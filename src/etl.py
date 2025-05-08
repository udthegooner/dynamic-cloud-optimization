from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, row_number, ceil, count
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType, DoubleType

# Borg cluster 2011 trace starts at 1900 EDT
TRACE_START_OFFSET = 600_000_000  # 600 seconds in µs
FILTERED_TASK_START_TIMESTAMP = 18_000_000_000 + TRACE_START_OFFSET # monday 0000 EDT
FILTERED_TASK_END_TIMESTAMP = 104_400_000_000 + TRACE_START_OFFSET # tues 0000 EDT

# Window size = 3 hours in microseconds
WINDOW_SIZE_US = 10_800_000_000

# use t3a.xlarge EC2 instance as reference for scaling normalized CPU and mem values
REF_CPU = 4   # cores
REF_MEM = 16  # GB

def main():
    spark = SparkSession.builder.appName("CleanClusterData").getOrCreate()

    task_events_schema = StructType([
        StructField("timestamp", LongType(), True),
        StructField("missing_info", StringType(), True),
        StructField("job_id", LongType(), True),
        StructField("task_index", IntegerType(), True),
        StructField("machine_id", IntegerType(), True),
        StructField("event_type", IntegerType(), True),
        StructField("username", StringType(), True),
        StructField("class", IntegerType(), True),
        StructField("priority", IntegerType(), True),
        StructField("cpu", DoubleType(), True),
        StructField("memory", DoubleType(), True),
        StructField("disk", DoubleType(), True),
        StructField("diff_machine_constraint", IntegerType(), True),
    ])

    df = spark.read.csv("../data/raw_task_events/*.csv.gz", header=False, schema=task_events_schema)

    # drop unecessary columns
    df = df.drop("task_index", "machine_id", "username", "class", "priority", "diff_machine_constraint")
    
    # filter tasks on submission event and without missing info
    tasks = df.filter((df["event_type"] == 0) & (df['missing_info'].isNull() | (df['missing_info'] == '')))

    # drop extra cols and rows with null values
    tasks = tasks.drop("event_type", "missing_info").dropna()

    # columns kept: ["timestamp", "cpu", "memory", "disk", "job_id"]

    # filter tasks in chosen time range
    tasks = tasks.filter((col("timestamp") >= FILTERED_TASK_START_TIMESTAMP) & 
                         (col("timestamp") < FILTERED_TASK_END_TIMESTAMP))
        
    # filter tasks with cpu and memory > 0
    tasks = tasks.filter((col("cpu") > 0) & (col("memory") > 0))

    # keep only most compute intensive task of each job
    # calculate compute_score as normalized cpu + memory
    tasks = tasks.withColumn("compute_score", col("cpu") + col("memory"))

    window = Window.partitionBy("job_id").orderBy(col("compute_score").desc())

    tasks = tasks.withColumn("rn", row_number().over(window)) \
                .filter(col("rn") == 1) \
                .drop("rn", "compute_score", "job_id")


    # Compute window_id = (timestamp - start_offset) // window_size
    tasks = tasks.withColumn(
                "window_id",
                ((col("timestamp") - lit(FILTERED_TASK_START_TIMESTAMP)) / lit(WINDOW_SIZE_US)).cast("integer")
            ).drop("timestamp")
    
    tasks = tasks.withColumn("cpu_cores", ceil(col("cpu") * lit(REF_CPU))) \
                .withColumn("mem_gb", col("memory") * lit(REF_MEM))
    
    # keep top 1% of resource heavy tasks in each window
    # Define window partition by window_id and order by cpu_cores, mem_gb descending
    partition_window = Window.partitionBy("window_id").orderBy(col("cpu_cores").desc(), col("mem_gb").desc())

    # Add row number per window
    tasks_ranked = tasks.withColumn("rank", row_number().over(partition_window))

    # Get total count per window_id
    window_counts = tasks.groupBy("window_id").agg(count("*").alias("total_tasks"))

    # Join counts back to tasks_ranked
    tasks_ranked = tasks_ranked.join(window_counts, on="window_id")

    # Compute threshold rank (top 1%)
    tasks_ranked = tasks_ranked.withColumn("threshold_rank", ceil(col("total_tasks") * lit(0.01)).cast("int"))

    # Filter to top 1%
    filtered_tasks = tasks_ranked.filter(col("rank") <= col("threshold_rank"))

    # Drop helper cols
    tasks = filtered_tasks.drop("rank", "total_tasks", "threshold_rank")

    tasks.persist()

    # LOAD
    # Get list of distinct window_ids
    window_ids = [row['window_id'] for row in tasks.select("window_id").distinct().collect()]
    window_ids.sort()

    output_dir = "../data/processed_tasks_by_window"

    for window_id in window_ids:
        # Filter for current window
        window_df = tasks.filter(col("window_id") == window_id)
        
        # Coalesce to 1 partition
        window_df = window_df.coalesce(1)
        print(f"Tasks in Window {window_id}: {window_df.count()}")
        
        # Write CSV
        output_path = f"{output_dir}/window_{window_id}.csv"
        window_df.write.csv(output_path, header=True, mode="overwrite")

    tasks.unpersist()

    spark.stop()

if __name__ == "__main__":
    main()
