from pyspark.sql import SparkSession
from pysparl.sql.functions import col, lit, row_number
from pyspark.sql.window import Window

# Borg cluster 2011 trace starts at 1900 EDT
TRACE_START_OFFSET = 600_000_000  # 600 seconds in µs
FILTERED_TASK_START_TIMESTAMP = 18_000_000_000 + TRACE_START_OFFSET # monday 0000 EDT
FILTERED_TASK_END_TIMESTAMP = 104_400_000_000 + TRACE_START_OFFSET # tues 0000 EDT

def main():
    spark = SparkSession.builder.appName("CleanClusterData").getOrCreate()

    # EXTRACT
    input_df = spark.read.csv("../data/part-00000-of-00500.csv.gz", header=False, inferSchema=True)

    # TRANSFORM

    # rename columns
    task_events_headers = ["timestamp", "missing_info", "job_id", "task_index", "machine_id",
                            "event_type", "username", "class", "priority", "cpu", "memory", 
                            "disk", "diff_machine_constraint"]
    df = input_df.toDF(*task_events_headers)

    # drop unecessary columns
    df = df.drop("job_id", "task_index", "machine_id",
                              "username", "class", "priority", "diff_machine_constraint")
    
    # filter tasks on submission event and without missing info
    tasks = df.filter((df["event_type"] == 0) & (df['missing_info'].isNull() | (df['missing_info'] == '')))

    # drop extra cols and rows with null values
    tasks = tasks.drop("event_type", "missing_info").dropna()

    # columns kept: ["timestamp", "cpu", "memory", "disk"]

    # filter tasks in chosen time range
    tasks = tasks.filter((col("timestamp") >= FILTERED_TASK_START_TIMESTAMP) & 
                         (col("timestamp") < FILTERED_TASK_END_TIMESTAMP))
    
    # filter tasks with cpu and memory > 0
    tasks = tasks.filter((col("cpu") > 0) & (col("memory") > 0))

    # add a row number to each task
    window = Window.orderBy("timestamp")
    tasks = tasks.withColumn("task_id", row_number().over(window))

    # LOAD
    print("Number of tasks:", tasks.count())
    tasks.write.csv("../data/processed_tasks.csv", header=True, mode="overwrite")

    spark.stop()

if __name__ == "__main__":
    main()




    
