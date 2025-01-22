# Final_project
Customer conversion prediction - Insurance

**Project Overview**
This project aims to predict whether a customer will subscribe to an insurance policy based on various attributes of the customer and details of marketing interactions. Using machine learning techniques, this solution analyzes marketing call data to make accurate predictions on customer subscriptions, ultimately helping insurance companies improve their marketing strategy and customer conversion rates.

**Problem Statement**
The dataset consists of information about marketing calls made to potential customers, with the goal of predicting whether a customer will subscribe to an insurance policy (target variable: y). Features such as age, job type, marital status, educational qualification, call details, and the outcome of previous marketing campaigns are provided for each customer.

**Key Objectives**
Data Preprocessing:
Clean the data, handle missing values, and encode categorical variables.
Scale or normalize features where necessary.
Exploratory Data Analysis (EDA):
Understand the distribution of features and identify patterns.
Explore relationships between features and the target variable to gain insights.
Dataset Balancing:
Assess whether the target variable is balanced.
Address imbalance issues, if present, using appropriate techniques.
Model Building:
Build and train various machine learning models for classification:
Logistic Regression
Decision Trees
Random Forest
Gradient Boosting
Train a clustering model for customer segmentation based on different customer attributes.
Hyperparameter Tuning:
Use cross-validation and techniques like Grid Search or Random Search to find the optimal hyperparameters for the models.
Model Evaluation:
Evaluate the models using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC to assess performance.
Select the best-performing model based on evaluation results.
Feature Importance:
Identify key features that influence customer subscription decisions.
Model Deployment:
Deploy the final model for real-time customer conversion prediction.
Save the model using Pickle for future use.

**Technologies Used**
Python: Primary language used for data analysis, preprocessing, machine learning, and model deployment.
Pandas & NumPy: Data manipulation and numerical operations.
Scikit-learn: Machine learning library for classification models, hyperparameter tuning, and evaluation.
Matplotlib & Seaborn: Libraries for data visualization and exploratory analysis.
Pickle: Used for serializing and saving the trained model.
Jupyter Notebooks: For iterative analysis and model development.

**How to Use**
Input Customer Data: Enter customer details such as age, job type, marital status, and call details.
Get Subscription Prediction: The trained model will predict whether the customer is likely to subscribe to an insurance policy (YES or NO).
Customer Segmentation: The clustering model can segment customers into different groups based on their attributes.

**Conclusion**
This project demonstrates the use of machine learning techniques to predict customer subscription behavior in the insurance industry. By analyzing customer data and marketing interactions, this model helps insurance companies optimize their marketing strategies, improve conversion rates, and make informed business decisions.
