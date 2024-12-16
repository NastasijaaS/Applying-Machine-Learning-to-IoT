import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructField, StructType, StringType, FloatType, DoubleType, IntegerType

fcd_topic = 'minnhen-fcd'
emission_topic = 'minnhen-emission'
kafka_url =  'kafka:9092'
save_kmeans_model = 'hdfs://namenode:9000/models/kmeans_model'
save_lr_model = 'hdfs://namenode:9000/models/lr_model'
output_path_clustered = 'hdfs://namenode:9000/output/claster'
output_path_predicted = 'hdfs://namenode:9000/output/predicted'
appName = "StreamingApp"


vehicleSchema = StructType([
        StructField("timestep_time", FloatType()),
        StructField("vehicle_angle", FloatType()),
        StructField("vehicle_id", IntegerType()),
        StructField("vehicle_lane", StringType()),
        StructField("vehicle_pos", FloatType()),
        StructField("vehicle_slope", FloatType()),
        StructField("vehicle_speed", FloatType()),
        StructField("vehicle_type", StringType()),
        StructField("vehicle_x", FloatType()),
        StructField("vehicle_y", FloatType())
    ])

emissionSchema = StructType([
        StructField("timestep_time", FloatType()),
        StructField("vehicle_CO", FloatType()),
        StructField("vehicle_CO2", FloatType()),
        StructField("vehicle_HC", FloatType()),
        StructField("vehicle_NOx", FloatType()),
        StructField("vehicle_PMx", FloatType()),
        StructField("vehicle_angle", FloatType()),
        StructField("vehicle_eclass", StringType()),
        StructField("vehicle_electricity", FloatType()),
        StructField("vehicle_id", IntegerType()),
        StructField("vehicle_lane", StringType()),
        StructField("vehicle_fuel", FloatType()),
        StructField("vehicle_noise", FloatType()),
        StructField("vehicle_pos", FloatType()),
        StructField("vehicle_route", StringType()),
        StructField("vehicle_speed", FloatType()),
        StructField("vehicle_type", StringType()),
        StructField("vehicle_waiting", FloatType()),
        StructField("vehicle_x", FloatType()),
        StructField("vehicle_y", FloatType())
    ])

if __name__ == "__main__":

    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).appName(appName).master("spark://spark:7077").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    cluster_model = PipelineModel.load(save_kmeans_model)
    regression_model = PipelineModel.load(save_lr_model)
    
    dfFcd = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_url) \
        .option("subscribe", fcd_topic) \
        .load()
    
    dfFcd.printSchema()

    dfEmission = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_url) \
        .option("subscribe", emission_topic) \
        .load()
    
    dfEmission.printSchema()
   
    dfFcd = dfFcd.selectExpr("timestamp as event_time", "CAST(value AS STRING)")
    dfFcd = dfFcd.withColumn("value", from_json(dfFcd["value"], schema=vehicleSchema))
    dfFcd = dfFcd.selectExpr(
        "value.vehicle_x as vehicle_x",
        "value.vehicle_y as vehicle_y",
    )

    dfEmission = dfEmission.selectExpr("timestamp as event_time", "CAST(value AS STRING)")
    dfEmission = dfEmission.withColumn("value", from_json(dfEmission["value"], schema=emissionSchema))
    dfEmission = dfEmission.selectExpr(
        "value.vehicle_fuel as vehicle_fuel",
        "value.vehicle_speed as vehicle_speed",
        "value.vehicle_noise as vehicle_noise",
        "value.vehicle_NOx as vehicle_NOx"
    )

    clustered_df = cluster_model.transform(dfFcd)
    predicted_df = regression_model.transform(dfEmission)

    predicted_df = predicted_df.select(["vehicle_fuel", "vehicle_speed", "vehicle_noise", "vehicle_NOx", "emission_prediction"])
    clustered_df = clustered_df.select(["vehicle_x", "vehicle_y", "congestion_cluster"])

    query_clustered = clustered_df \
        .writeStream \
        .format("csv") \
        .option("path", output_path_clustered) \
        .option("checkpointLocation", "checkpoint_dir") \
        .outputMode("append") \
        .start()

    query_predicted = predicted_df \
        .writeStream \
        .format("csv") \
        .option("path", output_path_predicted) \
        .option("checkpointLocation", "checkpoint_dir2") \
        .outputMode("append") \
        .start()

    query_clustered.awaitTermination()
    query_predicted.awaitTermination()

    spark.stop()