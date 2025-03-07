import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from chapters.overview import df

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.title("Data Exploration")

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show Basic Info
st.subheader("Basic Information")
st.write(f"Total Records: **{df.shape[0]}**")
st.write(f"Total Features: **{df.shape[1]}**")

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe(include='all'))

# Check for Missing Values
st.subheader("Missing Values")
missing_values = df.isnull().sum()
st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

# Visualizations
st.subheader("Data Visualizations")

# Gender Distribution
st.subheader("Gender Distribution")
gender_counts = df['Gender'].value_counts()
st.bar_chart(gender_counts)

# Depression Status Distribution
st.subheader("Depression Status Distribution")
st.bar_chart(df["Depression"].value_counts())

# CGPA Distribution
st.subheader("CGPA Distribution")
fig, ax = plt.subplots()
sns.histplot(df["CGPA"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Sleep Duration Distribution
st.subheader("Sleep Duration Distribution")
st.bar_chart(df["Sleep Duration"].value_counts())

# Work/Study Hours Distribution
st.subheader("Work/Study Hours Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Work/Study Hours"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
num_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
