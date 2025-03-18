import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Set Streamlit page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("assets/ModelTraining.png", use_container_width=True)

# Page Title
st.title("Hyperparameter Tuning")

# Load dataset
df = pd.read_csv("data/FeatureEngineeredData.csv")

# Display data (optional)
st.write(df.head())

# ///////////////////////////////////////////////////////////Prepare Data////////////////////////////

# Select features (X) and target (y)
X = df.drop('Depression', axis=1)  # Replace 'Depression' with your target column name if different
y = df['Depression']  # Replace 'Depression' with your target column name if different

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ///////////////////////////////////////////////////////////Define and Train the Model////////////////////////////

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)

# Train the model
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Display performance metrics before tuning as a table
metrics = {
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred)],
    "Recall": [recall_score(y_test, y_pred)],
    "F1 Score": [f1_score(y_test, y_pred)]
}

# Convert metrics to a DataFrame and display it as a table
metrics_df = pd.DataFrame(metrics)
st.subheader("Logistic Regression Model Performance")
st.table(metrics_df)

# Explanation text
st.write("As we can see, the performance of the model is currently based on the default parameters. "
         "Now, let's proceed with hyperparameter tuning to optimize the model's performance.")

# ///////////////////////////////////////////////////////////Hyperparameter Tuning with RandomizedSearchCV////////////////////////////

# Define the parameter distribution for RandomizedSearchCV
param_distributions = {
    'C': np.logspace(-3, 3, 10),  # Regularization strength (exponential scale)
    'solver': ['lbfgs', 'liblinear', 'saga'],  # Optimization algorithm
    'max_iter': [10000]  # Maximum number of iterations
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=log_reg, 
    param_distributions=param_distributions, 
    n_iter=20,  # Number of random combinations to try
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1, 
    random_state=42
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best model from the random search
best_log_reg = random_search.best_estimator_

# Predict using the tuned model
y_pred_tuned = best_log_reg.predict(X_test)

# Display performance metrics after tuning
st.subheader("Tuning Logistic Regression Model")

st.write("""
    In this step, we will perform hyperparameter tuning on the Logistic Regression model using **RandomizedSearchCV**.
    Unlike GridSearchCV, which tests all possible combinations, **RandomizedSearchCV** randomly selects a subset of 
    hyperparameter combinations to test. This method is more efficient for large search spaces and provides 
    near-optimal results faster.

    The key hyperparameters that we will tune include:
    - **C**: The regularization strength, controlling the trade-off between model complexity and generalization. We are testing values on an exponential scale.
    - **solver**: The algorithm used to optimize the model. We will experiment with different solvers like 'lbfgs', 'liblinear', and 'saga'.
    - **max_iter**: The maximum number of iterations for optimization, set to 10,000 to ensure proper convergence.

    After selecting the best hyperparameters, we will evaluate the model again and compare it with the baseline performance.
    This will help us determine the effectiveness of hyperparameter tuning.
""")

# Metrics after tuning
metrics_tuned = {
    "Accuracy": [accuracy_score(y_test, y_pred_tuned)],
    "Precision": [precision_score(y_test, y_pred_tuned)],
    "Recall": [recall_score(y_test, y_pred_tuned)],
    "F1 Score": [f1_score(y_test, y_pred_tuned)]
}


metrics_df_tuned = pd.DataFrame(metrics_tuned)
st.table(metrics_df_tuned)

st.write("Best Hyperparameters found: ", random_search.best_params_)
