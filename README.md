# NYC Taxi Trip Duration - Big Data Analysis 🚖

Conducted a big data analysis on NYC taxi trip records using **Apache Spark** for distributed computing. The project analyzes a large dataset and develops machine learning models to predict trip duration based on key input features.

---

## 📋 I. Introduction

New York City (NYC) is one of the largest cities globally, with a population of approximately 18 million people. Due to its dense population, traffic congestion is a persistent issue. Taxis are a popular mode of transportation within the city. In this project, we focused on predicting taxi trip durations using the **NYC Taxi Trip Duration dataset** to help users estimate travel times and plan trips more effectively.

The dataset, released by the NYC Taxi and Limousine Commission, contains more than 510,000 records, including pickup times, geographic coordinates, passenger counts, and other variables. Many machine learning models have been built on this dataset as part of a Kaggle challenge, but their accuracy has room for improvement. Most models use TensorFlow and convolutional neural networks. In contrast, we explored data manipulation and model development using **Scikit-learn** and **PySpark**, aiming to enhance accuracy and reliability.

We created visualizations to explore relationships between trip duration and other variables, analyzing patterns in trip count, distance, and duration across different weekdays and hours. Additionally, we built three machine learning models for insights and trip duration predictions: **Linear Regression**, **Gradient-Boosted Trees**, and **K-Means Clustering**.

---

## 🛠️ II. Data Preprocessing

Data preparation involved several key steps:

- Each column's data type was converted to appropriate formats (integer, double, string, or timestamp)
- Computed **Euclidean distance** between pickup and drop-off coordinates for all trips
- Filtered out trip durations outside the range of **3 minutes to 2 hours** (typical NYC taxi range)
- Excluded trips beyond **60 kilometers**, as taxis typically avoid long-distance routes
- Removed records with **null values** and invalid GPS coordinates
- Extracted features: **week_day**, **hour of day**, and **distance**

---

## 📊 III. Data Visualization

### Dataset Overview

Started with comprehensive dataset exploration including row count, column types, schema, and basic statistics. This helps identify the data distribution and any anomalies before modeling.

### Total Trips and Passenger Count by Day of Week

Total trips peak on **Fridays and Saturdays**, while average passenger counts are highest on **Sundays**, likely due to family outings. This pattern reflects how NYC residents and visitors use taxis differently across the week.

![Total Trips and Avg Passenger by Weekday](Documents/Total%20trips%20and%20Avg.%20Passenger%20count%20on%20Different%20days%20of%20week.png)

### Mean Trip Duration and Distance by Day of Week

Trip durations are generally shorter on weekends, reflecting reduced traffic, though distances tend to be **longer for leisure travel** on Saturdays and Sundays.

![Mean Trip Duration and Distance by Weekday](Documents/Mean%20Trip%20Duration%20and%20Distance%20on%20Different%20days%20of%20week.png)

### Trip Duration Distribution & Outlier Analysis

A multi-panel visualization showing trip duration distribution on both **linear and log scales**, a **boxplot for outlier detection**, and the **passenger count distribution**. The log-scale view reveals the underlying distribution shape that the linear scale hides due to extreme outliers.

![Trip Duration Distribution and Outlier Analysis](Documents/Trip%20Duration%20Distribution%2C%20log%20scale%2C%20Boxplot%28Outlier%20Detection%29%2C%20Passenger%20Count%20Distribution.png)

### Hourly Demand Patterns

Number of trips per hour and average trip duration by hour. Peak demand occurs during **evening rush hours (6-9 PM)**, and rush-hour effects are clearly visible in the duration patterns — trips take longer during high-traffic periods.

![Hourly Trip Patterns](Documents/Number%20of%20trips%20by%20per%20hour%20%2C%20average%20trip%20duration%20by%20hour.png)

### Distance vs. Trip Duration Relationship

Distance is a **key predictor** of trip duration. Approximately **70% of trips cover less than 10 kilometers** and take under 2000 seconds (~30 minutes). The relationship is most visible on a log-log scale, where it appears linear.

![Distance vs Trip Duration](Documents/Distance%20and%20Trip%20Duration.png)

---

## 🧠 IV. Algorithms

Outlier removal was crucial for effective modeling. Categorical variables like weekdays were converted to indexed columns using **StringIndexer**, and **VectorAssembler** compiled features into vectors for regression models. The target column was `trip_duration`. Data was split **70% for training and 30% for testing**.

### Regression Models

**Linear Regression**: Based on the equation `y = mx + c`, built using PySpark's ML library. Hyperparameters such as `regParam`, `elasticNetParam`, `maxIter`, and `fitIntercept` were optimized using `ParamGridBuilder` and `CrossValidator` with **5-fold cross-validation**.

**Gradient-Boosted Trees Regression**: A powerful ensemble algorithm that minimizes prediction errors by iteratively adjusting residuals. We tuned `maxDepth`, `maxIter`, and `maxBins` for optimal performance — achieving significantly higher accuracy than linear regression.

### K-Means Clustering

Using scikit-learn's `KMeans`, we identified geographic and behavioral patterns by grouping similar trips. The **Elbow method** determined the optimal number of clusters (**k = 4**).

![Elbow Method](Documents/elbow%20method.png)

After determining the optimal k, we analyzed the feature distribution per cluster to understand what defines each group of trips:

![Feature Distribution per Cluster](Documents/Feature%20Distribution%20per%20cluster.png)

---

## 🎯 V. Results

### Linear Regression
- **Accuracy**: ~50.23%
- **RMSE**: 500.76 seconds

Optimized parameters provided a baseline performance. Linear Regression struggles with the non-linear relationships in taxi data.

### Gradient-Boosted Trees Regression
- **Accuracy**: ~75%
- **RMSE**: 362.1 seconds

Gradient-Boosted Trees **significantly outperformed** Linear Regression by capturing non-linear patterns and feature interactions.

### K-Means Clustering

Despite challenges in cluster differentiation, we identified meaningful patterns. The radar plot below visualizes the key characteristics of each cluster:

![Cluster Characteristics](Documents/cluster%20characteristics.png)

---

## 💬 VI. Discussion

While 75% accuracy from the Gradient-Boosted Trees model shows room for improvement, it's promising given hardware limitations. Longer training times restricted hyperparameter tuning, emphasizing the need for robust computational resources.

Key challenges:
- **Long training times** limited the hyperparameter search space
- **Memory constraints** required careful data sampling for visualization
- **Outliers** had to be carefully handled to avoid skewing model results

---

## 🔮 VII. Future Work

1. **Advanced Models**: Implement **XGBoost** and **LightGBM** with proper hyperparameter tuning using `RandomizedSearchCV`
2. **Cloud-Based Resources**: Utilize AWS or GCP for more extensive model tuning and scalability
3. **Real-Time Data Integration**: Incorporate live data through APIs and build continuous learning pipelines
4. **Geographic Features**: Add NYC taxi zone boundaries and neighborhood-based features
5. **Deployment**: Build a **Streamlit web app** for real-time trip duration prediction
6. **Time Series Forecasting**: Predict daily ride demand using Prophet or ARIMA

This project demonstrates valuable insights into NYC taxi trips and highlights opportunities for further development in predictive modeling for urban transportation.

---

## 🛠️ Tech Stack

- **Languages**: Python 3.10
- **Big Data**: Apache Spark (PySpark)
- **ML Libraries**: scikit-learn, PySpark MLlib
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Folium, Plotly
- **Environment**: Google Colab, Jupyter Notebooks
- **Version Control**: Git, GitHub

---

## 📂 Project Structure

```
NYC-Taxi-Duration-Data-Analysis/
├── Data/
│   └── Data Cleaning.ipynb
├── Preprocessing/
│   ├── NYC Taxi Duration Data Preprocessing.ipynb
│   └── NYC Taxi Duration Data Visualization.ipynb
├── ML Model/
│   ├── Final NYC Taxi Duration - Clustering.ipynb
│   └── Final NYC Taxi Trip Duration - Regression.ipynb
├── Documents/
│   └── [visualization images]
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. Clone this repository:
```bash
   git clone https://github.com/Aryan09092001/NYC-Taxi-Duration-Data-Analysis.git
```

2. Download the dataset from [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Run notebooks in order:
   - `Data/Data Cleaning.ipynb`
   - `Preprocessing/NYC Taxi Duration Data Preprocessing.ipynb`
   - `Preprocessing/NYC Taxi Duration Data Visualization.ipynb`
   - `ML Model/Final NYC Taxi Trip Duration - Regression.ipynb`
   - `ML Model/Final NYC Taxi Duration - Clustering.ipynb`

---



⭐ **If you found this project helpful, please consider giving it a star!**
