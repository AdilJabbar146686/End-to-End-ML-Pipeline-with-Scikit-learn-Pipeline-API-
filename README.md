Objective

The objective of this task is to build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset.

The pipeline must include data preprocessing, model training, hyperparameter tuning, and export for deployment.

Methodology / Approach

Data Loading

Train, validation, and test sets loaded from Telco Churn dataset.

Preprocessing

Used ColumnTransformer with:

StandardScaler for numerical features.

OneHotEncoder for categorical features.

Pipeline Construction

Built an end-to-end Pipeline that combines preprocessing + model.

Implemented two ML models:

Logistic Regression

Random Forest Classifier

Hyperparameter Tuning

Used GridSearchCV to optimize model hyperparameters.

Cross-validation (cv=5) ensures robust evaluation.

Model Evaluation

Evaluated models using accuracy score and classification report.

Compared performance on test dataset.

Exporting

Saved the final trained pipeline with joblib for reusability in production.

Key Results / Observations

Preprocessing with Pipeline ensured that scaling and encoding were seamlessly applied within the workflow.

Logistic Regression provided a solid baseline with interpretability.

Random Forest achieved better performance after hyperparameter tuning.

The final model was exported as a .pkl file, making it directly loadable for deployment.

Demonstrated production-readiness by ensuring:

Consistent preprocessing.

Easy model reuse.

Reproducibility via pipelines and joblib export.

Skills Gained

Building end-to-end ML pipelines using Scikit-learn.

Applying hyperparameter tuning with GridSearchCV.

Exporting and reusing models with joblib.

Designing ML workflows with production-readiness practices.
