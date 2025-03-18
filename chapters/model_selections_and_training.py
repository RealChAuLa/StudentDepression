import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report, roc_curve, auc
import time

# Set Streamlit page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("./assets/ModelTraining.png", use_container_width=True)

# Page Title
st.title("Model Selection and Training")


# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, Y_train, Y_test, model_name):
    # Training time measurement
    start_time = time.time()
    model.fit(X_train, Y_train)
    training_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Calculate metrics
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='weighted')
    recall = recall_score(Y_test, y_pred, average='weighted')
    f1 = f1_score(Y_test, y_pred, average='weighted')

    # Training metrics
    train_accuracy = accuracy_score(Y_train, y_train_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(Y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')

    # ROC curve data (for binary classification)
    if len(np.unique(Y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(Y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        except:
            fpr, tpr, roc_auc = None, None, None
    else:
        fpr, tpr, roc_auc = None, None, None

    # Return all metrics
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'training_time': training_time,
        'train_accuracy': train_accuracy,
        'roc_data': (fpr, tpr, roc_auc) if fpr is not None else None
    }


# Function to display evaluation results
def display_results(results, model_name):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Accuracy', 'CV Mean Accuracy'],
            'Value': [
                f"{results['accuracy']:.4f}",
                f"{results['precision']:.4f}",
                f"{results['recall']:.4f}",
                f"{results['f1_score']:.4f}",
                f"{results['train_accuracy']:.4f}",
                f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)
        st.write(f"Training Time: {results['training_time']:.4f} seconds")

    with col2:
        st.write("### Confusion Matrix")
        # Create a confusion matrix using Plotly
        cm = results['confusion_matrix']
        fig = px.imshow(cm,
                        labels=dict(x="Predicted Label", y="True Label", color="Count"),
                        x=['0', '1'],
                        y=['0', '1'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig.update_layout(title=f'Confusion Matrix - {model_name}')
        st.plotly_chart(fig)

    # ROC Curve for binary classification
    if results['roc_data'] is not None:
        fpr, tpr, roc_auc = results['roc_data']
        st.write("### ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig.update_layout(
            title='Receiver Operating Characteristic',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        st.plotly_chart(fig)

    # Classification report
    st.write("### Classification Report")
    report = classification_report(Y_test, results['model'].predict(X_test), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


# Sidebar for Algorithm Navigation
st.sidebar.title("Training Algorithms")
algorithm_links = {
    "Logistic Regression": "#logistic-regression",
    "Decision Tree": "#decision-tree",
    "Support Vector Machine (SVM)": "#svm",
    "k-Nearest Neighbors (k-NN)": "#knn",
    "Random Forest": "#random-forest",
    "Neural Networks (MLP)": "#neural-networks"
}
for name, link in algorithm_links.items():
    st.sidebar.markdown(f"- [{name}]({link})")

# Load preprocessed dataset
df = pd.read_csv("./data/FeatureEngineeredData.csv")

# Display dataset info
st.write("### Dataset Information")
st.write(f"Dataset Shape: {df.shape}")
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())



col1, col2 = st.columns([1, 1])
# Target Variable distribution
with col1:
    st.write("### Target Variable Distribution")
    # Create target distribution chart with Altair
    target_counts = df['Depression'].value_counts().reset_index()
    target_counts.columns = ['Depression', 'Count']
    target_chart = alt.Chart(target_counts).mark_bar().encode(
        x=alt.X('Depression:N', title='Depression Status (0: No, 1: Yes)'),
        y=alt.Y('Count:Q', title='Number of Records'),
        color=alt.Color('Depression:N', scale=alt.Scale(scheme='reds')),
        tooltip=['Depression', 'Count']
    ).properties(
        title='Distribution of Depression Classes',
        width=600,
        height=400
    )
    st.altair_chart(target_chart, use_container_width=True)

    # Split data into features and target
    Y = df['Depression']
    X = df.drop(columns=['Depression'])  # Adjust target column name if different

# Train-Test Split
with col2:
    st.write("### Dataset Splitting")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.write(f"Training Set: {X_train.shape[0]} samples")
    st.write(f"Test Set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled features
st.write("### Scaled Features")
st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head())

# Initialize dictionary to store model results
all_results = {}

# Logistic Regression
st.header("Logistic Regression", anchor="logistic-regression")
st.write("A simple baseline model that predicts depression status using a linear decision boundary.")

with st.expander("Logistic Regression Details"):
    st.write("""
    Logistic Regression is a statistical method for analyzing a dataset where the outcome is categorical. 
    It works by estimating the probabilities using a logistic function.
    """)

    # Training the model
    lr_model = LogisticRegression(random_state=42)
    lr_results = evaluate_model(lr_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "Logistic Regression")
    all_results['Logistic Regression'] = lr_results

    # Display results
    display_results(lr_results, "Logistic Regression")

    # Feature importance
    if hasattr(lr_model, 'coef_'):
        st.write("### Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': lr_model.coef_[0]
        }).sort_values('Coefficient', ascending=False)

        # Create feature importance chart with Plotly
        fig = px.bar(coef_df,
                     x='Coefficient',
                     y='Feature',
                     orientation='h',
                     title='Logistic Regression Coefficients')
        st.plotly_chart(fig)

# Decision Tree
st.header("Decision Tree", anchor="decision-tree")
st.write("A tree-based model that learns decision rules to classify depression status.")

with st.expander("Decision Tree Details"):
    st.write("""
    Decision Tree is a flowchart-like tree structure where each internal node denotes a test on an attribute, 
    each branch represents an outcome of the test, and each leaf node holds a class label.
    """)

    # Training the model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_results = evaluate_model(dt_model, X_train, X_test, Y_train, Y_test, "Decision Tree")
    all_results['Decision Tree'] = dt_results

    # Display results
    display_results(dt_results, "Decision Tree")

    # Feature importance
    st.write("### Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create feature importance chart with Plotly
    fig = px.bar(feature_imp,
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 title='Decision Tree Feature Importance')
    st.plotly_chart(fig)


# k-Nearest Neighbors
st.header("k-Nearest Neighbors (k-NN)", anchor="knn")
st.write("A non-parametric algorithm that classifies based on the majority class of nearest neighbors.")

with st.expander("k-NN Details"):
    st.write("""
    k-Nearest Neighbors is a simple, instance-based learning algorithm that makes predictions based on the k closest training examples in the feature space.
    """)

    # Training the model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_results = evaluate_model(knn_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "k-NN")
    all_results['k-NN'] = knn_results

    # Display results
    display_results(knn_results, "k-Nearest Neighbors")

# Random Forest
st.header("Random Forest", anchor="random-forest")
st.write("An ensemble model of decision trees that improves accuracy and reduces overfitting.")

with st.expander("Random Forest Details"):
    st.write("""
    Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.
    """)

    # Training the model
    rf_model = RandomForestClassifier(random_state=42)
    rf_results = evaluate_model(rf_model, X_train, X_test, Y_train, Y_test, "Random Forest")
    all_results['Random Forest'] = rf_results

    # Display results
    display_results(rf_results, "Random Forest")

    # Feature importance
    st.write("### Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create feature importance chart with Plotly
    fig = px.bar(feature_imp,
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 title='Random Forest Feature Importance')
    st.plotly_chart(fig)

# Neural Networks (MLP)
st.header("Neural Networks (MLP)", anchor="neural-networks")
st.write("A multi-layer perceptron (MLP) that learns complex patterns in data.")

with st.expander("Neural Networks Details"):
    st.write("""
    Multi-layer Perceptron (MLP) is a class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.
    """)

    # Training the model
    mlp_model = MLPClassifier(random_state=42, max_iter=1000)
    mlp_results = evaluate_model(mlp_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "Neural Networks")
    all_results['Neural Networks'] = mlp_results

    # Display results
    display_results(mlp_results, "Neural Networks (MLP)")


# Support Vector Machine (SVM)
st.header("Support Vector Machine (SVM)", anchor="svm")
st.write("A powerful classifier that finds the optimal hyperplane for class separation.")

with st.expander("SVM Details"):
    st.write("""
    Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for both classification and regression tasks. 
    It finds the optimal hyperplane that best separates the classes.
    """)

    # Training the model
    svm_model = SVC(random_state=42, probability=True)
    svm_results = evaluate_model(svm_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "SVM")
    all_results['SVM'] = svm_results

    # Display results
    display_results(svm_results, "Support Vector Machine")



# Model Comparison
st.header("Model Comparison")

# Extract accuracy and training time
model_names = list(all_results.keys())
accuracies = [all_results[model]['accuracy'] for model in model_names]
train_times = [all_results[model]['training_time'] for model in model_names]

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Training Time (s)': train_times
}).sort_values('Accuracy', ascending=False)

st.dataframe(comparison_df)

#