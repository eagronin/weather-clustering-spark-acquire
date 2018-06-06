# Data Acquisition
This section imports and describes the dataset **minute-weather.csv**, which is subsequently used in cluster analysis using k-means to identify different weather patterns for a weather station in San Diego, CA.

The [next section](https://eagronin.github.io/weather-clustering-spark-prepare/) explores, cleans and scales the data to prepare them for clustering analysis in the subsequent section.

The file **minute-weather.csv** was downloaded from the Coursera website and saved on the Cloudera cloud.

This is a comma-separated file that contains weather data. The data comes from a weather station located in San Diego, CA. The weather station is equipped with sensors that capture weather-related measurements such as air temperature, air pressure, and relative humidity. Data was collected for a period of three years, from September 2011 to September 2014, to ensure that sufficient data for different seasons and weather conditions is captured.  Sensor measurements from the weather station were captured at one-minute intervals.  This is the same dataset that was used to construct **daily_weather.csv** described [here](https://eagronin.github.io/weather-classification-spark-acquire/).

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





# Data Preparation
This section explores, cleans and scales the data to prepare them for the cluster analysis that identifies different weather patterns for a weather station in San Diego, CA using k-means. 

The dataset is described and imported in the [previous section](https://eagronin.github.io/weather-clustering-spark-acquire/).

The analysis is described in the [next section](https://eagronin.github.io/weather-clustering-spark-analyze/).

The imported dataset includes over 1.5 million rows, as indicated by `df.count()`.  For the purpose of this analysis a smaller dataset was used that contains only one-tenth of the data.  The following code creates such a subset of data by keeping every 10th row in the subset and dropping all the other rows:

```python
filteredDF = df.filter((df.rowID % 10) == 0)
filteredDF.count()
```

The number of rows in the subset is 158,726.

Below are the summary statistics: 

```python
filteredDF.describe().toPandas().transpose()
```

```
summary			count	mean		stddev		min	max
rowID			158726	793625.0	458203.93	0	1587250
air_pressure		158726	916.83		3.05		905.0	929.5
air_temp		158726	61.85		11.83		31.64	99.5
avg_wind_direction	158680	162.15		95.27		0.0	359.0
avg_wind_speed		158680	2.77		2.05		0.0	31.9
max_wind_direction	158680	163.46		92.45		0.0	359.0
max_wind_speed		158680	3.40		2.41		0.1	36.0
rain_accumulation	158725	3.18E-4		0.01		0.0	3.12
rain_duration		158725	0.40		8.66		0.0	2960.0
relative_humidity	158726	47.60		26.21		0.9	93.0
```

The low average values for rain accumulation and duration in this dataset suggest that the data were collected during a dry period. The code below outputs the counts of days when the values of rain accumulation and duration are 0:

```python
filteredDF.filter(filteredDF.rain_accumulation == 0.0).count()
filteredDF.filter(filteredDF.rain_duration == 0.0).count()
```

For rain accumulation the count is 157,812 days, while for rain duration the count is 157,237 days, which are almost all the days in the sample.  Since the values for these features are almost all 0 (i.e., very limited variation in the data) and for the purpose of speeding up the analyses, these  features are dropped from the DataFrame. We also drop the hpwren_timestamp feature since it is not used in the analysis, as well as rowID since it is the row number:

```python
workingDF = filteredDF.drop('rain_accumulation').drop('rain_duration').drop('hpwren_timestamp').drop('rowID')
```

Next, we drop rows with missing values and count how many rows were dropped:

```python
before = workingDF.count()
workingDF = workingDF.na.drop()
after = workingDF.count()
before - after
```

The code above indicate that 46 rows in the `workingDF` dataframe had missing values in at least one feature, before these rows were dropped.  

Next, we combine the remaining features into a single vector column. Let's create an array of the columns we want to combine, and use VectorAssembler to create the vector column:

```python
 featuresUsed = ['air_pressure',
 'air_temp',
 'avg_wind_direction',
 'avg_wind_speed',
 'max_wind_direction',
 'max_wind_speed',
 'relative_humidity']
 
assembler = VectorAssembler(inputCols = featuresUsed, outputCol = 'features_unscaled')
assembled = assembler.transform(workingDF)
```

Finally, since the features are on different scales (e.g., air temperature ranges from 31.6 to 99.5, while air pressure ranges from 905.0 to 929.5), they need to be scaled. We will scale them using `StandardScaler()` so that each feature has the mean of 0 and the standard deviation of 1:

```
scaler = StandardScaler(inputCol = 'features_unscaled', outputCol = 'features', withStd = True, withMean = True)
scalerModel = scaler.fit(assembled)
scaledData = scalerModel.transform(assembled)
```

Next step: [Analysis](https://eagronin.github.io/weather-clustering-spark-analyze/)





# Analysis
This section identifies distinct weather patterns using k-means clustering of the data collected from a weather station in San Diego, CA . First, we determine the optimal number of clusters using an elbow plot.  Then we find cluster centers for the optimal number of clusters to identify weather patterns corresponding to each cluster.

The [previous section](https://eagronin.github.io/weather-clustering-spark-prepare/) explores, cleans and scales the data to prepare it for the clustering analysis.

The [next section](https://eagronin.github.io/weather-clustering-spark-report/) reports and interprets the results.

The k-means algorithm requires that the number of clusters (k) has to be specified. To determine a good value for k, we will use the “elbow” method. This method applies k-means using different values for k and calculating the within-cluster sum-of-squared error (WSSE).  This process can be compute-intensive, because k-means is applied multiple times. Therefore, for creating the elbow plot we use only a subset of the dataset. Specifically, we keep every third row from the dataset and drop all the other rows:

```python
from itertools import cycle, islice
from math import sqrt
from numpy import array
from pandas.tools.plotting import parallel_coordinates
from pyspark.ml.clustering import KMeans as KM
from pyspark.mllib.linalg import DenseVector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scaledData = scaledData.select('features', 'rowID')
elbowset = scaledData.filter((scaledData.rowID % 3) == 0).select('features')
elbowset.persist()
```

The persist() method in the last row is used to keep the resulting dataset in memory for faster processing.  

Next, we define several functions that we will need for creating the elbow plot and parallel coordinate plots of cluster centroids.

WSSE discussed above is calculated using the following function: 

```python
def computeCost(featuresAndPrediction, model):
    allClusterCenters = [DenseVector(c) for c in model.clusterCenters()]
    arrayCollection   = featuresAndPrediction.rdd.map(array)

    def error(point, predictedCluster):
        center = allClusterCenters[predictedCluster]
        z      = point - center
        return sqrt((z*z).sum())
    
    return arrayCollection.map(lambda row: error(row[0], row[1])).reduce(lambda x, y: x + y)
```

The function below iterates over the number of clusters to determine WSSE for each number of clusters to create the elbow plot:

```python
def elbow(elbowset, clusters):
	wsseList = []	
	for k in clusters:
		print("Training for cluster size {} ".format(k))
		kmeans = KM(k = k, seed = 1)
		model = kmeans.fit(elbowset)
		transformed = model.transform(elbowset)
		featuresAndPrediction = transformed.select("features", "prediction")

		W = computeCost(featuresAndPrediction, model)
		print("......................WSSE = {} ".format(W))

		wsseList.append(W)
	return wsseList
```

The function below plots WSSE against the number of clusters:

```python
def elbow_plot(wsseList, clusters):
	wsseDF = pd.DataFrame({'WSSE' : wsseList, 'k' : clusters })
	wsseDF.plot(y='WSSE', x='k', figsize=(15,10), grid=True, marker='o')
```

The following function converts cluster centers determined by the the k-means algorithm to a pandas dataframe in order to plot cluster centers in matplotlib (Spark dataframes cannot be plotted using matplotlib): 

```python
def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P
```

Finally, the function below plots cluster centers:

```python
def parallel_plot(data, P):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(P)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
```    
  
The code in the remainder of this section calls the functions above to create the elbow plot and fit the k-means algorithm for the chosen number of clusters.  

The following code calculates WSSE for each number of clusters ranging from 2 to 30:

```
clusters = range(2, 31)
wsseList = elbow(elbowset, clusters)
```

This code generates the following output, which is going to be used as input into the function that create elbow plot:

```
Training for cluster size 2 
......................WSSE = 114993.13 
Training for cluster size 3 
......................WSSE = 104181.09
Training for cluster size 4 
......................WSSE = 94577.27
Training for cluster size 5 
......................WSSE = 87993.46 
Training for cluster size 6 
......................WSSE = 85084.23 
Training for cluster size 7 
......................WSSE = 81664.96 
Training for cluster size 8 
......................WSSE = 78397.76 
Training for cluster size 9 
......................WSSE = 76599.60 
Training for cluster size 10 
......................WSSE = 74023.93 
Training for cluster size 11 
......................WSSE = 72772.61 
Training for cluster size 12 
......................WSSE = 70281.81 
Training for cluster size 13 
......................WSSE = 69473.53 
Training for cluster size 14 
......................WSSE = 68756.12 
Training for cluster size 15 
......................WSSE = 67394.28 
Training for cluster size 16 
......................WSSE = 66698.44 
Training for cluster size 17 
......................WSSE = 64559.11 
Training for cluster size 18 
......................WSSE = 63205.19 
Training for cluster size 19 
......................WSSE = 62368.485 
Training for cluster size 20 
......................WSSE = 62444.64 
Training for cluster size 21 
......................WSSE = 61265.25 
Training for cluster size 22 
......................WSSE = 60936.73 
Training for cluster size 23 
......................WSSE = 60109.64 
Training for cluster size 24 
......................WSSE = 60087.31 
Training for cluster size 25 
......................WSSE = 59084.88 
Training for cluster size 26 
......................WSSE = 58498.07 
Training for cluster size 27 
......................WSSE = 58215.79 
Training for cluster size 28 
......................WSSE = 58132.84 
Training for cluster size 29 
......................WSSE = 57426.52 
Training for cluster size 30 
......................WSSE = 57099.24 
```

By calling `elbow_plot(wsseList, clusters)` we create an elbowplot, which is presented in the [next section](https://eagronin.github.io/weather-clustering-spark-report/).

As we will discuss in the [next section](https://eagronin.github.io/weather-clustering-spark-report/), we choose the number of clusters to be 12 based on the elbow plot.

Once we have chosen the number of clusters using a scaled down version of the dataset, we will go back to the `scaledData` dataset to fit the k-means algorithm for 12 clusters using that dataset.  We again choose the persist() method to keep the datasest in memory for faster processing:

```python
scaledDataFeat = scaledData.select('features')
scaledDataFeat.persist()

kmeans = KMeans(k = 12, seed = 1)
model = kmeans.fit(scaledDataFeat)
transformed = model.transform(scaledDataFeat)

transformed.head(10)
```

The code above results in a dataset that contains in each row a point in the feature sapce as the first column and the cluster ID to which this point belongs as the second column:

```
[Row(features=DenseVector([-1.4846, 0.2454, -0.6839, -0.7656, -0.6215, -0.7444, 0.4923]), prediction=2),
 Row(features=DenseVector([-1.4846, 0.0325, -0.1906, -0.7656, 0.0383, -0.6617, -0.3471]), prediction=0),
 Row(features=DenseVector([-1.5173, 0.1237, -0.6524, -0.3768, -0.4485, -0.3723, 0.4084]), prediction=2),
 Row(features=DenseVector([-1.5173, 0.0629, -0.7468, -0.3768, -0.654, -0.4137, 0.3931]), prediction=2),
 Row(features=DenseVector([-1.5173, 0.1846, -0.8518, -0.0852, -0.8162, -0.2069, 0.3741]), prediction=2),
 Row(features=DenseVector([-1.5501, 0.1542, -0.6314, -0.7656, -0.4809, -0.7857, 0.1451]), prediction=2),
 Row(features=DenseVector([-1.5829, 0.1846, -0.8308, -1.0085, -0.6756, -1.0338, 0.1451]), prediction=2),
 Row(features=DenseVector([-1.6156, 0.1998, -0.8413, -0.3768, -0.7189, -0.4137, 0.5572]), prediction=2),
 Row(features=DenseVector([-1.6156, -0.0132, -0.9987, 0.255, -1.0109, 0.0411, 0.9121]), prediction=2),
 Row(features=DenseVector([-1.6156, -0.0436, -0.9987, 0.4008, -0.9568, 0.3305, 0.9502]), prediction=2)]
 ...
```

The fitted model is then used to determine the center measurement of each cluster:

```python
centers = model.clusterCenters()
centers
```

These cluster centers are shown below:

```
[array([-0.13720796,  0.6061152 ,  0.22970948, -0.62174454,  0.40604553, -0.63465994, -0.42215364]),
 array([ 1.42238994, -0.10953198, -1.10891543, -0.07335197, -0.96904335, -0.05226062, -0.99615617]),
 array([-0.63637648,  0.01435705, -1.1038928 , -0.58676582, -0.96998758, -0.61362174,  0.33603011]),
 array([-0.22385278, -1.06643622,  0.5104215 , -0.24620591,  0.68999967, -0.24399706,  1.26206479]),
 array([ 1.17896517, -0.25134204, -1.15089838,  2.11902126, -1.04950228,  2.23439263, -1.12861666]),
 array([-1.14087425, -0.979473  ,  0.42483303,  1.68904662,  0.52550171,  1.65795704,  1.03863542]),
 array([ 0.50746307, -1.08840683, -1.20882766, -0.57604727, -1.0367013 , -0.58206904,  0.97099067]),
 array([ 0.14064028,  0.83834618,  1.89291279, -0.62970435, -1.54598923, -0.55625032, -0.75082891]),
 array([-0.0339489 ,  0.98719067, -1.33032244, -0.57824562, -1.18095582, -0.58893358, -0.81187427]),
 array([-0.22747944,  0.59239924,  0.40531475,  0.6721331 ,  0.51459992,  0.61355559, -0.15474261]),
 array([ 0.3334222 , -0.99822761,  1.8584392 , -0.68367089, -1.53246714, -0.59099434,  0.91004892]),
 array([ 0.3051367 ,  0.67973831,  1.36434828, -0.63793718,  1.631528  , -0.58807924, -0.67531539])]
```

It is difficult to compare the cluster centers by just looking at these numbers. Therefore, we will visualize these numbers using parallel coordinates plots, which are used to visualize multi-dimensional data.  Each line plots the centroid of a cluster, and all of the features are plotted together on the same chart. Because the feature values were scaled to have mean = 0 and standard deviation = 1, the values on the y-axis of these parallel coordinates plots show the number of standard deviations from the mean.

The plots are created with matplotlib using a Pandas DataFrame.  Each row in the dataframe contains the cluster center coordinates and cluster label. We use the `pd_centers()` function described above to create the Pandas DataFrame:
 
```python
P = pd_centers(featuresUsed, centers)
```

Next step: [Results](https://eagronin.github.io/weather-clustering-spark-report/)





# Results
This section visualizes and discusses the results of the k-means clustering analysis identifying different weather patterns for a weather station in San Diego, CA. The elbowplot discussed in the previous section is shown first.  Then we plot and discuss several charts showing cluster centers for the optimal number of clusters to identify distinct weather patterns.

The analysis is discussed in the [previous section](https://eagronin.github.io/weather-clustering-spark-analyze/).

Let's first plot the elbowplot by calling elbow_plot() function described in the [previous section](https://eagronin.github.io/weather-clustering-spark-analyze/).

![](elbow plot)

The values for the number of clusters (k) are plotted against WSSE values.  The chart shows that WSSE declines as the number of clusters increases.  This is intuitive, because allowing for a larger number of clusters facilitates shorter distances between each point in the feature space and cluster centers.  However, as the number of clusters increases, the curve gradually flattnes, as the contribution to the decline in WSSE by adding a cluster diminishes. The bend in the curve (or elbow) provides an estimate for the optimal value for k.  In this plot, we see that the elbow in the curve is between 10 and 15, so let's choose k = 12. The subsequent plots of cluster centers are all based on the number of clusters k = 12.

First, we determine weather patterns on "dry days" (characterized by low relative humidity):

```python
parallel_plot(P[P['relative_humidity'] < -0.5], P)
```

![](chart 1)

While all the clusters shown on the chart above have low relative humidity, the other features, such as air tempreature and wind direction are different across these clusters.  Therefore, the clusters represent distinct weather patterns, all of which are characterized by low relative humidity.

For example, cluster 4 has low wind direction (whcih means that the samples in this cluster have wind coming from the N and NE directions) and high wind speed.  On the opposite end of the spectrum is cluster 11, characterized by high wind direction and low wind speed.

Second, we determine weather patterns on "warm days" (characterized by high air temperature): 

```python
parallel_plot(P[P['air_temp'] > 0.5], P)
```

![](chart 2)

While all clusters in this plot are characterized by realtively high air temperature, they differ in values for other features, thereby representing distinct patterns of warm weather.  

Third, we deterimne weather patterns on "cool days", such as weather samples with high relative humidity and low air temperature:

```python
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)], P)
```

![](chart 3)

For example, cluster 5 represets samples with high wind speed which, in the presence of cool temperature and high relative humidity suggest stormy weather patterns.

The only cluster that has not been included on the three previous charts is cluster 2:

```python
parallel_plot(P.iloc[[2]], P)
```

![](chart 4)

We can see that air temperature, relative humidity and wind speed for this cluster all fall within one standard deviation from the mean, suggesting a mild weather pattern.

Previous step: [Analysis](https://eagronin.github.io/weather-clustering-spark-analyze/)
