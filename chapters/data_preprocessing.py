import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    file_path = "data/Student_Depression_Dataset.csv"
    df = pd.read_csv(file_path)
    return df

st.title("Missing Value Handling & Verification")

df = load_data()

st.subheader("❌ Missing Value Count (Before Handling)")
missing_before = df.isnull().sum()
missing_table_before = pd.DataFrame({"Column": missing_before.index, "Missing Values": missing_before.values})
st.write(missing_table_before[missing_table_before["Missing Values"] > 0])

df_cleaned = df.copy()

for col in df_cleaned.select_dtypes(include=[np.number]):  
    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)


df_cleaned.dropna(inplace=True)

st.subheader("✅ Missing Value Count (After Handling)")
missing_after = df_cleaned.isnull().sum()
missing_table_after = pd.DataFrame({"Column": missing_after.index, "Missing Values": missing_after.values})
st.write(missing_table_after[missing_table_after["Missing Values"] > 0] if missing_after.sum() > 0 else "No missing values left!")


if 'Sleep Duration' in df_cleaned.columns:
    st.subheader("🛏️ Sleep Duration Categories and Counts")
    sleep_duration_counts = df_cleaned['Sleep Duration'].value_counts()
    st.write(sleep_duration_counts)
else:
    st.write("The 'Sleep Duration' column does not exist in the dataset.")

if 'Dietary Habits' in df_cleaned.columns:
    st.subheader("🍽️ Dietary Habits Categories and Counts")
    dietary_habits_counts = df_cleaned['Dietary Habits'].value_counts()
    st.write(dietary_habits_counts)
else:
    st.write("The 'Dietary Habits' column does not exist in the dataset.")


if 'Degree' in df_cleaned.columns:
    st.subheader("🎓 Degree Categories and Row Counts")
    degree_counts = df_cleaned['Degree'].value_counts()
    st.write(degree_counts)
else:
    st.write("The 'Degree' column does not exist in the dataset.")


