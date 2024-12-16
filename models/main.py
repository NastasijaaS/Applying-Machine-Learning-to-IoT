import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression

data_location = 'hdfs://namenode:9000/dir/data.csv'
save_kmeans_model = 'hdfs://namenode:9000/models/kmeans_model'
save_lr_model = 'hdfs://namenode:9000/models/lr_model'

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("ModelCreation") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    data = spark.read.option("header", "true").option("delimiter", ";").csv(data_location, inferSchema=True)

    (trainData, testData) = data.randomSplit([0.7, 0.3], seed=42)

    fcd_features = ["vehicle_x", "vehicle_y"]
    
    kmeans_assembler = VectorAssembler(inputCols=fcd_features, outputCol="fcd_features")
  
    kmeans = KMeans(seed=42, featuresCol="fcd_features", predictionCol="congestion_cluster")
    kmeans_pipeline = Pipeline(stages=[kmeans_assembler, kmeans])
    param_grid_kmeans = ParamGridBuilder() \
        .addGrid(kmeans.k, [4, 5, 6, 7, 8]) \
        .build()

    kmeans_evaluator = ClusteringEvaluator(predictionCol="congestion_cluster", featuresCol="fcd_features", metricName="silhouette")

    kmeans_cv = CrossValidator(estimator=kmeans_pipeline, \
                               estimatorParamMaps=param_grid_kmeans, \
                               evaluator=kmeans_evaluator, \
                               numFolds=3)

    kmeans_model = kmeans_cv.fit(trainData)
    kmeans_predictions = kmeans_model.bestModel.transform(testData)

    silhouette_score = kmeans_evaluator.evaluate(kmeans_predictions)
    print(f"Best K-Means Silhouette Score: {silhouette_score}")
    kmeans_model.bestModel.write().overwrite().save(save_kmeans_model)

    regression_features = ["vehicle_fuel", "vehicle_speed", "vehicle_noise"]
    regression_assembler = VectorAssembler(inputCols=regression_features, outputCol="regression_features")

    lr = LinearRegression(featuresCol="regression_features", labelCol="vehicle_NOx", predictionCol="emission_prediction", standardization=True)

    lr_pipeline = Pipeline(stages=[regression_assembler, lr])

    param_grid_lr = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    lr_evaluator = RegressionEvaluator(labelCol="vehicle_NOx", predictionCol="emission_prediction", metricName="rmse")

    lr_cv = CrossValidator(estimator=lr_pipeline, \
                           estimatorParamMaps=param_grid_lr, \
                           evaluator=lr_evaluator, \
                           numFolds=3)

    lr_model = lr_cv.fit(trainData)
    lr_predictions = lr_model.bestModel.transform(testData)

    rmse_score = lr_evaluator.evaluate(lr_predictions)
    print(f"Best Linear Regression RMSE: {rmse_score}")
    
    lr_model.bestModel.write().overwrite().save(save_lr_model)

    spark.stop()
