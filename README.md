End-to-End ML Pipeline with Scikit-learn Pipeline API

This repository demonstrates a complete End-to-End Machine Learning Pipeline built using the Scikit-learn Pipeline API.
The project focuses on building a robust, reusable, and scalable ML workflow — from data preprocessing and model training to evaluation and model persistence.

Project Overview

This project showcases how to streamline the machine learning process using Scikit-learn’s Pipeline and ColumnTransformer tools.
It is designed to ensure reproducibility, cleaner code, and easier model deployment.

The primary goal of this project is to:

Handle data preprocessing automatically

Train and evaluate a predictive model

Save and reuse the trained model using .pkl format

Demonstrate professional ML project structuring

Problem Statement

Predict customer churn based on available features using machine learning.
The pipeline includes data transformation, encoding, feature scaling, and model training steps — all wrapped neatly inside the Scikit-learn Pipeline framework.

Technologies and Libraries Used

Python 3.x

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

Joblib / Pickle

Repository Structure
End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API-/
│
├── task2.ipynb                 # Main Jupyter Notebook containing the pipeline code
├── best_churn_model.pkl        # Trained machine learning model
├── README.md                   # Project documentation

⚙️ Key Features

Data preprocessing with automatic handling of categorical and numerical features
End-to-End Pipeline using Pipeline() and ColumnTransformer()
Model training, testing, and evaluation
Model serialization using .pkl for deployment
Well-structured and modular workflow

Model Workflow

Data Import & Exploration
Load and inspect the dataset.

Data Preprocessing

Handle missing values

Encode categorical variables

Scale numerical features

Pipeline Construction
Combine all preprocessing and model steps using Pipeline() and ColumnTransformer().

Model Training & Evaluation
Train the model on training data and evaluate using test data.

Model Saving
Export the final trained model as best_churn_model.pkl.

How to Use

Clone the repository

git clone https://github.com/AdilJabbar146686/End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API-.git
cd End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API-


Install dependencies

pip install -r requirements.txt


Run the Jupyter Notebook

jupyter notebook task2.ipynb


Load the trained model

import pickle

with open('best_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

Results

The pipeline achieved strong predictive performance with optimized preprocessing and feature scaling.

Model performance metrics (accuracy, precision, recall, F1-score) are evaluated and visualized in the notebook.

Future Improvements

Integrate with Flask or Streamlit for web-based model deployment

Add hyperparameter tuning using GridSearchCV

Incorporate more advanced models (e.g., XGBoost, LightGBM)

Automate data ingestion and logging

Author
Adil Jabbar
https://github.com/AdilJabbar146686/End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API-.git
