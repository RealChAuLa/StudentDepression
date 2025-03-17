import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Set Streamlit page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("assets/ModelTraining.png", use_container_width=True)

# Page Title
st.title("Model Selection and Training")

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
df = pd.read_csv("data/FeatureEngineeredData.csv")

# Split data into features and target
X = df.drop(columns=['Depression'])  # Adjust target column name if different
y = df['Depression']

# Train-Test Split
st.write("### Dataset Splitting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training Set: {X_train.shape[0]} samples")
st.write(f"Test Set: {X_test.shape[0]} samples")

# Logistic Regression
st.header("Logistic Regression", anchor="logistic-regression")
st.write("A simple baseline model that predicts depression status using a linear decision boundary.")

# Decision Tree
st.header("Decision Tree", anchor="decision-tree")
st.write("A tree-based model that learns decision rules to classify depression status.")

# Support Vector Machine (SVM)
st.header("Support Vector Machine (SVM)", anchor="svm")
st.write("A powerful classifier that finds the optimal hyperplane for class separation.")

# k-Nearest Neighbors (k-NN)
st.header("k-Nearest Neighbors (k-NN)", anchor="knn")
st.write("A non-parametric algorithm that classifies based on the majority class of nearest neighbors.")

# Random Forest
st.header("Random Forest", anchor="random-forest")
st.write("An ensemble model of decision trees that improves accuracy and reduces overfitting.")

# Neural Networks (MLP)
st.header("Neural Networks (MLP)", anchor="neural-networks")
st.write("A multi-layer perceptron (MLP) that learns complex patterns in data.")

# Final Note
st.success("Select an algorithm from the sidebar to learn more and proceed with training.")
