import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessed data
from chapters.data_preprocessing import df_cleaned

# Set page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("assets/feature_engineering.png", use_container_width=True)

# Page Title
st.title("Feature Engineering")

# Check available columns before proceeding
st.write("### Columns in Dataset:")
st.write(df_cleaned.columns.tolist())  

st.write("### Preview of the Data:")
st.write(df_cleaned.head())

# Profession Refinement
st.subheader("Profession Refinement")

if 'Profession' in df_cleaned.columns:
    st.write("""
    After filtering, only the profession **'Student'** remains in the dataset. 
    Since all other professions had very few records and were removed, the **'Profession'** column is no longer needed.
    """)

    # Drop the 'Profession' column
    df_cleaned.drop(columns=['Profession'], inplace=True)

    # Display message with emphasis
    st.markdown("""
        <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
            <b style="color:#b30000;">The 'Profession' column has been removed
            since only 'Student' remains as the profession.</b> 
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("The 'Profession' column is not found in the dataset. It may have been removed earlier.")
