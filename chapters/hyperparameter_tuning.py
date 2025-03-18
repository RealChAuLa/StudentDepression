import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import time
from skopt import BayesSearchCV

# Set Streamlit page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("assets/hyperparameter-tuning.jpg", use_container_width=True)

# Page Title
st.title("Hyperparameter Tuning")

# Load dataset
df = pd.read_csv("data/FeatureEngineeredData.csv")

# Prepare Data
X = df.drop('Depression', axis=1)
Y = df['Depression']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load the model and scaler
model = joblib.load('models/Logistic Regression.pkl')
scaler = joblib.load('models/scaler.pkl')
X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)

# Baseline Performance
Y_pred = model.predict(X_test_scaled)
metrics = {
    "Accuracy": [accuracy_score(Y_test, Y_pred)],
    "Precision": [precision_score(Y_test, Y_pred)],
    "Recall": [recall_score(Y_test, Y_pred)],
    "F1 Score": [f1_score(Y_test, Y_pred)]
}
metrics_df = pd.DataFrame(metrics)


# Function to display performance comparison
def display_comparison(metrics_dfs, method_names):
    comparison_df = pd.DataFrame()

    for i, (df, name) in enumerate(zip(metrics_dfs, method_names)):
        df_copy = df.copy()
        df_copy.insert(0, 'Method', name)
        comparison_df = pd.concat([comparison_df, df_copy], ignore_index=True)

    # Create bar chart for comparison
    fig = px.bar(comparison_df, x='Method', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                 barmode='group', title='Model Performance Comparison')
    st.plotly_chart(fig, use_container_width=True)

    # Display table
    st.subheader("Performance Metrics Comparison")
    st.table(comparison_df)


# Original model performance
st.header("Original Model Performance")
st.table(metrics_df)

st.write("Let's proceed with different hyperparameter tuning techniques to optimize the model's performance.")

# Initialize list to store results for comparison
all_metrics_dfs = [metrics_df]
method_names = ["Original Model"]
tuned_models = {}

# RandomizedSearchCV
st.subheader("Randomized Search")
with st.expander("RandomizedSearchCV Tuning"):
    st.write("""
    RandomizedSearchCV samples a fixed number of hyperparameter combinations from specified distributions.
    Unlike GridSearchCV, it doesn't test all combinations, making it more efficient for large search spaces.

    **Advantages:**
    - More efficient than grid search for large parameter spaces
    - Can find good solutions with fewer evaluations
    - Works well when some parameters are more important than others

    **Disadvantages:**
    - May miss optimal combinations by chance
    - Less exhaustive than grid search
    """)

    # Define the parameter distribution
    param_distributions = {
        'C': np.logspace(-3, 3, 10),
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [10000]
    }

    start_time = time.time()

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Fit RandomizedSearchCV
    with st.spinner('Running Randomized Search...'):
        random_search.fit(X_train_scaled, Y_train)

    end_time = time.time()

    # Get the best model
    best_random = random_search.best_estimator_
    tuned_models['RandomizedSearchCV'] = best_random

    # Predict and evaluate
    y_pred_random = best_random.predict(X_test_scaled)
    metrics_random = {
        "Accuracy": [accuracy_score(Y_test, y_pred_random)],
        "Precision": [precision_score(Y_test, y_pred_random)],
        "Recall": [recall_score(Y_test, y_pred_random)],
        "F1 Score": [f1_score(Y_test, y_pred_random)]
    }
    metrics_df_random = pd.DataFrame(metrics_random)
    all_metrics_dfs.append(metrics_df_random)
    method_names.append("RandomizedSearchCV")

    st.subheader("RandomizedSearchCV Results")
    st.table(metrics_df_random)
    st.write(f"Best Hyperparameters found: {random_search.best_params_}")
    st.write(f"Time taken: {end_time - start_time:.2f} seconds")

# GridSearchCV
st.subheader("Grid Search")
with st.expander("GridSearchCV Tuning"):
    st.write("""
    GridSearchCV exhaustively generates candidates from a grid of parameter values.
    It tests all possible combinations of parameters, guaranteeing to find the best combination within the specified grid.

    **Advantages:**
    - Guaranteed to find the optimal combination within the given parameter grid
    - Systematic and thorough exploration of parameter space
    - Simple to understand and implement

    **Disadvantages:**
    - Computationally expensive for large parameter spaces
    - "Curse of dimensionality" - exponential growth of combinations
    - Inefficient when some parameters are more important than others
    """)

    # Define smaller parameter grid for demonstration (to avoid long computation)
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [10000]
    }

    start_time = time.time()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit GridSearchCV
    with st.spinner('Running Grid Search...'):
        grid_search.fit(X_train_scaled, Y_train)

    end_time = time.time()

    # Get the best model
    best_grid = grid_search.best_estimator_
    tuned_models['GridSearchCV'] = best_grid

    # Predict and evaluate
    y_pred_grid = best_grid.predict(X_test_scaled)
    metrics_grid = {
        "Accuracy": [accuracy_score(Y_test, y_pred_grid)],
        "Precision": [precision_score(Y_test, y_pred_grid)],
        "Recall": [recall_score(Y_test, y_pred_grid)],
        "F1 Score": [f1_score(Y_test, y_pred_grid)]
    }
    metrics_df_grid = pd.DataFrame(metrics_grid)
    all_metrics_dfs.append(metrics_df_grid)
    method_names.append("GridSearchCV")

    st.subheader("GridSearchCV Results")
    st.table(metrics_df_grid)
    st.write(f"Best Hyperparameters found: {grid_search.best_params_}")
    st.write(f"Time taken: {end_time - start_time:.2f} seconds")

# Bayesian Optimization
st.subheader("Bayesian Optimization")
with st.expander("Bayesian Optimization Tuning"):
    st.write("""
    Bayesian Optimization uses past evaluation results to choose the next set of hyperparameters to evaluate.
    It builds a probabilistic model of the objective function and uses it to select the most promising points to evaluate.

    **Advantages:**
    - More efficient than random or grid search
    - Learns from previous evaluations to guide the search
    - Often finds better solutions with fewer evaluations
    - Well-suited for expensive-to-evaluate functions

    **Disadvantages:**
    - More complex to implement and understand
    - May get stuck in local optima
    - Requires careful selection of prior and acquisition function
    """)

    try:
        # Define the search space for Bayesian optimization
        search_spaces = {
            'C': (1e-3, 1e3, 'log-uniform'),
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [10000]
        }

        start_time = time.time()

        # Set up BayesSearchCV
        bayes_search = BayesSearchCV(
            estimator=LogisticRegression(random_state=42),
            search_spaces=search_spaces,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        # Fit BayesSearchCV
        with st.spinner('Running Bayesian Optimization...'):
            bayes_search.fit(X_train_scaled, Y_train)

        end_time = time.time()

        # Get the best model
        best_bayes = bayes_search.best_estimator_
        tuned_models['BayesSearchCV'] = best_bayes

        # Predict and evaluate
        y_pred_bayes = best_bayes.predict(X_test_scaled)
        metrics_bayes = {
            "Accuracy": [accuracy_score(Y_test, y_pred_bayes)],
            "Precision": [precision_score(Y_test, y_pred_bayes)],
            "Recall": [recall_score(Y_test, y_pred_bayes)],
            "F1 Score": [f1_score(Y_test, y_pred_bayes)]
        }
        metrics_df_bayes = pd.DataFrame(metrics_bayes)
        all_metrics_dfs.append(metrics_df_bayes)
        method_names.append("BayesSearchCV")

        st.subheader("BayesSearchCV Results")
        st.table(metrics_df_bayes)
        st.write(f"Best Hyperparameters found: {bayes_search.best_params_}")
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        st.error(f"Error in Bayesian Optimization: {e}")
        st.write(
            "Bayesian Optimization requires installing 'scikit-optimize'. If you want to use this method, install it with: pip install scikit-optimize")

# Model Comparison
st.header("Model Comparison")
display_comparison(all_metrics_dfs, method_names)

# Find the best method
best_method_idx = 0
best_accuracy = 0

for i, df in enumerate(all_metrics_dfs):
    if df["Accuracy"][0] > best_accuracy:
        best_accuracy = df["Accuracy"][0]
        best_method_idx = i

best_method = method_names[best_method_idx]
st.success(f"The best performing method is: {best_method} with accuracy: {best_accuracy:.4f}")

# Save best model
if best_method != "Original Model":
    best_tuned_model = tuned_models[best_method]

    # Create a directory for saved models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the best model to a file
    model_filename = f'models/best_tuned_model_{best_method}.pkl'
    joblib.dump(best_tuned_model, model_filename)

    st.success(f"Best model has been saved to {model_filename}")

    # Add a download button
    with open(model_filename, 'rb') as f:
        model_bytes = f.read()
        st.download_button(
            label="Download Best Tuned Model",
            data=model_bytes,
            file_name=f"best_tuned_model_{best_method}.pkl",
            mime="application/octet-stream"
        )