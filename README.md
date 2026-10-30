# ML_Project 
# Airline Passenger Satisfaction Prediction

## About the Project

Airline Passenger Satisfaction Prediction is a Machine Learning project that predicts whether an airline passenger is satisfied or neutral/dissatisfied based on their travel details and experience.

The project uses passenger information such as age, gender, customer type, travel class, flight distance, delays and different in-flight service ratings.

We trained and compared multiple Machine Learning models to identify which algorithm gives the best prediction results.

The main aim is to understand the factors that affect passenger satisfaction and help airlines improve their services.

## Problem Statement

Passenger satisfaction is an important factor for airlines. Different aspects of a journey, such as seat comfort, online boarding, in-flight Wi-Fi, food quality and delays, can affect the overall passenger experience.

Manually analysing a large number of passenger responses is difficult.

To solve this problem, we developed a Machine Learning system that analyses passenger data and predicts satisfaction based on the available features.

## Dataset Information

The dataset contains airline passenger records with information about their travel experience.

- Total Records: 129,880
- Original Columns: 24
- Predictive Features: 23
- Target Variable: Satisfaction

The target variable contains two categories:

- Satisfied
- Neutral or Dissatisfied

### Dataset Distribution

| Category | Passengers | Percentage |
|---|---:|---:|
| Neutral/Dissatisfied | 73,452 | 56.55% |
| Satisfied | 56,428 | 43.45% |

## Project Workflow

The project follows these steps:

1. Dataset Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Correlation Analysis
6. Feature Engineering
7. Data Splitting
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Final Model Selection
12. Passenger Satisfaction Prediction

## Data Preprocessing

Before training the models, we performed several preprocessing steps.

- **Missing Value Handling:** Missing values in arrival delay and survey ratings were handled using median imputation.
- **Data Cleaning:** Removed unnecessary passenger ID columns.
- **Outlier Handling:** Examined unusual values in flight distance and delay features.
- **Feature Transformation:** Applied log transformation to flight distance.
- **Feature Engineering:** Combined departure and arrival delays into a single Total Delay feature.
- **Age Grouping:** Divided passenger ages into different groups.
- **Service Scores:** Created combined Ground Service and In-flight Service scores.
- **Encoding:** Converted categorical features into numerical values.
- **Feature Scaling:** Applied scaling where required.

After feature engineering, the dataset contained 29 features.

## Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify patterns.

Some of the analysis included:

- Distribution of satisfied and dissatisfied passengers.
- Analysis of passenger age and gender.
- Relationship between travel class and satisfaction.
- Analysis of flight distance and delays.
- Comparison of different service ratings.
- Correlation heatmap to understand relationships between features.

### Correlation Analysis

A correlation heatmap was created to identify relationships between different features.

It helped us understand how strongly the features were related and identify highly correlated variables.

Departure Delay and Arrival Delay showed a high correlation of approximately 0.965, so they were combined into Total Delay.

## Data Splitting

The dataset was divided into three parts using stratified splitting.

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 70% | 90,916 |
| Validation | 15% | 19,482 |
| Testing | 15% | 19,482 |

- Training data was used to train the models.
- Validation data was used to evaluate and tune model performance.
- Testing data was used to check the final performance on unseen data.

## Machine Learning Algorithms

We compared three Machine Learning algorithms.

### 1. Random Forest

Random Forest is an ensemble learning algorithm that uses multiple decision trees.

Each tree makes its own prediction, and the final result is decided through majority voting.

### 2. XGBoost

XGBoost stands for Extreme Gradient Boosting.

It builds decision trees sequentially, where each new tree tries to correct the errors made by previous trees.

### 3. LightGBM

LightGBM stands for Light Gradient Boosting Machine.

It is a gradient boosting algorithm that uses histogram-based learning and leaf-wise tree growth to improve efficiency and prediction performance.

## Model Performance

We compared the three models using testing accuracy and other evaluation metrics.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| LightGBM | 96.16% | 96.72% | 94.35% | 95.52% |
| XGBoost | 95.97% | 96.37% | 94.28% | 95.31% |
| Random Forest | 95.17% | 95.58% | 93.19% | 94.37% |

### Final Model Selection

LightGBM was selected as the final model because it achieved the highest testing accuracy of 96.16%.

It also provided a good balance between precision, recall and F1 score.

## Model Evaluation

The following evaluation metrics were used:

- **Accuracy:** Measures the percentage of correct predictions.
- **Precision:** Measures how many predicted positive results were actually positive.
- **Recall:** Measures how many actual positive cases were correctly identified.
- **F1 Score:** Combines precision and recall into one score.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes.
- **Confusion Matrix:** Shows correct and incorrect predictions.

### Training and Testing Accuracy

Training and testing accuracy were compared to understand how well the models performed on unseen data.

| Model | Training Accuracy | Testing Accuracy |
|---|---:|---:|
| LightGBM | 98.11% | 96.16% |
| XGBoost | 98.53% | 95.97% |
| Random Forest | 98.66% | 95.17% |

The difference between training and testing accuracy was relatively small for all three models, indicating good generalisation.

## Feature Importance

Feature importance was analysed to understand which factors contributed most to passenger satisfaction predictions.

Some of the important features were:

- Online Boarding
- In-flight Wi-Fi Service
- Travel Class
- Type of Travel
- Other passenger service ratings

Online Boarding and In-flight Wi-Fi Service were among the most influential features.

## User Interface

We developed a frontend using React.js to make the prediction system easier to use.

The interface allows users to enter passenger details and service ratings and view the predicted satisfaction result.

### Frontend Technologies

- React.js
- HTML
- CSS
- JavaScript

The frontend communicates with the backend through API requests to obtain predictions.

## Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and model evaluation |
| LightGBM | Classification model |
| XGBoost | Classification model |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| React.js | Frontend development |
| HTML | Web page structure |
| CSS | UI styling |
| JavaScript | Frontend functionality |

## Future Scope

The project can be improved in several ways:

- Use additional passenger feedback data.
- Improve model performance through further tuning.
- Add more features related to passenger experience.
- Deploy the application for practical use.
- Provide more detailed insights into passenger satisfaction.
- Help airlines identify areas that need service improvement.

## Conclusion

This project demonstrates how Machine Learning can be used to predict airline passenger satisfaction using passenger details and service ratings.

We implemented and compared Random Forest, XGBoost and LightGBM. Among these models, LightGBM achieved the best testing accuracy of 96.16%.

The system can help airlines understand passenger experiences, identify dissatisfied customers and improve their overall service quality.

## Project Information

**Project Name:** Airline Passenger Satisfaction Prediction

**Domain:** Machine Learning

**Problem Type:** Binary Classification

**Best Performing Model:** LightGBM

**Testing Accuracy:** 96.16%

**Frontend:** React.js

**Programming Language:** Python

---

Developed as part of an academic Machine Learning project.
