import streamlit as st
import pandas as pd

df_preprocessed = pd.read_csv("data/PreprocessedData.csv")

# Set page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("https://raw.githubusercontent.com/RealChAuLa/StudentDepression/master/assets/feature_engineering.png", use_container_width=True)

# Page Title
st.title("Feature Engineering")

# Check available columns before proceeding
st.write("### Columns in Dataset:")
st.write(df_preprocessed.columns.tolist())

st.write("### Preview of the Data:")
st.write(df_preprocessed.head())


######################### Column Id #####################


st.header("Column 'id'")

if 'id' in df_preprocessed.columns:
    st.write("""
    The **'id'** column is a unique identifier for each record in the dataset. 
    It is not useful for the model and will be removed.
    """)

    # Drop the 'id' column
    df_preprocessed.drop(columns=['id'], inplace=True)

    # Display message with emphasis
    st.markdown("""
        <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
            <b style="color:#b30000;">The 'id' column has been removed since it is not useful for the model.</b> 
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("The 'id' column is not found in the dataset. It may have been removed earlier.")




#################### Column Profession



st.header("Column 'Profession'")

if 'Profession' in df_preprocessed.columns:
    st.write("""
    After filtering, only the profession **'Student'** remains in the dataset. 
    Since all other professions had very few records and were removed, the **'Profession'** column is no longer needed.
    """)

    # Drop the 'Profession' column
    df_preprocessed.drop(columns=['Profession'], inplace=True)

    # Display message with emphasis
    st.markdown("""
        <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
            <b style="color:#b30000;">The 'Profession' column has been removed
            since only 'Student' remains as the profession.</b> 
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("The 'Profession' column is not found in the dataset. It may have been removed earlier.")


##################### Column Work Pressure ####################


st.header("Column 'Work Pressure'")
if 'Work Pressure' in df_preprocessed.columns:
    st.write("""
    After filtering, only the category **'0'** remains in the dataset. 
    Since all other categories had very few records and were removed, the **'Work Pressure'** column is no longer needed.
    """)

    # Drop the 'Work Pressure' column
    df_preprocessed.drop(columns=['Work Pressure'], inplace=True)

    # Display message with emphasis
    st.markdown("""
        <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
            <b style="color:#b30000;">The 'Work Pressure' column has been removed
            since only '0' remains as the category.</b> 
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("The 'Work Pressure' column is not found in the dataset. It may have been removed earlier.")



##################### Column Job Satisfaction ####################



st.header("Column 'Job Satisfaction'")
if 'Job Satisfaction' in df_preprocessed.columns:
    st.write("""
    After filtering, only the category **'0'** remains in the dataset. 
    Since all other categories had very few records and were removed, the **'Job Satisfaction'** column is no longer needed.
    """)

    # Drop the 'Job Satisfaction' column
    df_preprocessed.drop(columns=['Job Satisfaction'], inplace=True)

    # Display message with emphasis
    st.markdown("""
        <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
            <b style="color:#b30000;">The 'Job Satisfaction' column has been removed
            since only '0' remains as the category.</b> 
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("The 'Job Satisfaction' column is not found in the dataset. It may have been removed earlier.")



#############Feature Engineered Data####################
st.write("### Preview of the Feature Engineered Data:")
st.write(df_preprocessed.head(5))

#Export Feature Engineered Data
df_preprocessed.to_csv("data/FeatureEngineeredData.csv", index=False)


