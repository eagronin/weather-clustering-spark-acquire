# Data Acquisition

## Overview

This section describes and imports the **minute-weather.csv** dataset, which is subsequently used in cluster analysis using k-means to identify different weather patterns for a weather station in San Diego, CA.

The [next section](https://eagronin.github.io/weather-clustering-spark-prepare/) explores, cleans and scales the data to prepare them for clustering analysis in the subsequent section.

## Data

The file **minute-weather.csv** was downloaded from the Coursera website and saved on the Cloudera cloud.

This is a comma-separated file that contains weather data. The data come from a weather station located in San Diego, CA. The weather station is equipped with sensors that capture weather-related measurements such as air temperature, air pressure, and relative humidity. The data was collected for a period of three years, from September 2011 to September 2014, to ensure that sufficient data for different seasons and weather conditions is captured.  Sensor measurements from the weather station were captured at one-minute intervals.  This is the same dataset that was used to construct **daily_weather.csv** described [here](https://eagronin.github.io/weather-classification-spark-acquire/).

The following code imports **daily_weather.csv** from a folder on the cloud:

```python
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from notebooks import utils
%matplotlib inline


sqlContext = SQLContext(sc)
df = sqlContext.read.load('file:///home/cloudera/Downloads/big-data-4/minute_weather.csv', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')
```                          

Next step: [Data Preparation](https://eagronin.github.io/weather-clustering-spark-prepare/)
