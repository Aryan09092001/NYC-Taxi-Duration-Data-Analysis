**NYC Taxi Trip Duration - Big Data Analysis**

Conducted a big data analysis on NYC taxi trip records using a Spark cluster. The project aimed to analyze a large dataset utilizing the distributed computing framework Apache Spark and develop a machine learning model to predict trip duration based on key input features.

**I. Introduction**

New York City (NYC) is one of the largest cities globally, with a population of approximately 18 million people. Due to its dense population, traffic congestion is a persistent issue. Taxis are a popular mode of transportation within the city. In this project, we focused on predicting taxi trip durations using the "NYC taxi trip duration dataset," accessible here, to help users estimate travel times and plan trips more effectively.

The dataset, released by the NYC Taxi and Limousine Commission, contains more than 510,000 records, including pickup times, geographic coordinates, passenger counts, and other variables. Many machine learning models have been built on this dataset as part of a Kaggle challenge, but their accuracy has room for improvement. Most models use TensorFlow and convolutional neural networks. In contrast, we explored data manipulation and model development using Scikit-learn and PySpark, aiming to enhance accuracy and reliability.

We created visualizations to explore relationships between trip duration and other variables, analyzing patterns in trip count, distance, and duration across different weekdays. Additionally, we built three machine learning models for insights and trip duration predictions: linear regression, gradient-boosted trees, and K-means clustering. Below, we discuss our data preprocessing approach.

**II. Data Preprocessing**

Data preparation involved several key steps. Each column’s data type was converted to appropriate formats such as integer, double, string, or timestamp. We computed distances between pickup and drop-off coordinates for all trips. Assuming typical NYC taxi trips do not exceed two hours, we filtered out durations outside the range of 3 minutes to 2 hours. Similarly, trips beyond 60 kilometers were excluded, as taxis typically avoid long-distance routes. Additional preprocessing steps were implemented to refine the dataset for modeling.

![Preprocessing Diagram](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.001.png)

![Data Cleanup](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.002.png)

**III. Data Visualization**

We created various visualizations to extract insights from the data. For example, total trips peak on Fridays and Saturdays, while average passenger counts are highest on Sundays, likely due to family outings. Trip durations are generally shorter on weekends, reflecting reduced traffic, though distances tend to be longer for leisure travel.

![Weekly Patterns](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.003.png)

![Trip Duration Trends](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.004.png)

Distance is a key variable for trip duration predictions. Our analysis shows that approximately 70% of trips cover less than 10 kilometers and take under 2000 seconds (about 30 minutes). This compact distribution suggests outliers, as a few trips exhibit unusually high durations and distances.

![Distance vs. Duration](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.005.png)

**IV. Algorithms**

Outlier removal was crucial for effective modeling. We converted categorical variables like weekdays into indexed columns and used VectorAssembler to compile features into vectors for regression models. The target column was trip\_duration. The data was split into 70% for training and 30% for testing.

**I. Regression Models**

**Linear Regression**: This model, based on the equation y = mx + c, was built using PySpark’s ML library. Hyperparameters such as regParam, elasticNetParam, maxIter, and fitIntercept were optimized using ParamGridBuilder and CrossValidator.

**Gradient-Boosted Trees Regression**: A powerful supervised learning algorithm that minimizes prediction errors by iteratively adjusting residuals. We tuned maxDepth, maxIter, and maxBins for better performance, achieving higher accuracy than linear regression.

**II. K-Means Clustering**

Using scikit-learn’s KMeans, we identified patterns by grouping similar data points. The Elbow method determined the optimal number of clusters (k = 4). Cluster analysis revealed feature influences on trip duration, with additional box plots and radar charts providing a clear view of distribution patterns.

![Elbow Method](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.006.png)

![Cluster Analysis](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.007.png)

**V. Results**

**I. Linear Regression**

Optimized parameters yielded 50.23% accuracy with an RMSE of 500.76. Below are the best parameters, coefficients, and intercept for the linear model:

![Linear Regression Results](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.008.png)

**II. Gradient-Boosted Trees Regression**

This model achieved approximately 75% accuracy with an RMSE of 362.1. Gradient-boosted trees significantly outperformed linear regression in predictive power.

![GBT Results](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.009.png)

**III. K-Means Clustering**

Despite challenges in cluster differentiation, we identified meaningful patterns and generated a radar plot to visualize key cluster characteristics:

![Radar Plot](./Documents/Aspose.Words.aba6c92a-e782-445b-b727-2cf5eb96087d.010.png)

**VI. Discussion**

While 75% accuracy from the gradient-boosted trees model shows room for improvement, it is promising given hardware limitations. Longer training times restricted hyperparameter tuning, emphasizing the need for robust computational resources.

**Future Work**

1. **Cloud-Based Resources**: Utilizing AWS or GCP will allow for more extensive model tuning and scalability.
2. **Real-Time Data Integration**: Incorporating live data through APIs and building continuous learning pipelines will enhance predictive accuracy over time.

This project demonstrates valuable insights into NYC taxi trips and highlights opportunities for further development in predictive modeling for urban transportation.
