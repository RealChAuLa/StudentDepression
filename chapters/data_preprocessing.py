import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

def load_data():
    file_path = "data/Student_Depression_Dataset.csv"
    df = pd.read_csv(file_path)
    return df

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.image("./assets/3.jpg", use_container_width=True)
st.title("Data Preprocessing")

st.write(
    "Data preprocessing is important because raw data is often messy, missing, or inconsistent. Without cleaning and preparing the data, models may give incorrect results. Preprocessing helps by fixing missing values, removing errors, and making the data easier for models to understand, which improves accuracy and performance."
)

df = load_data()




####################################### HANDLING MISSING VALUE #####################

st.subheader("Missing Data Analysis")
missing_before = df.isnull().sum()
missing_table_before = pd.DataFrame({"Column": missing_before.index, "Missing Values": missing_before.values})
st.write(missing_table_before[missing_table_before["Missing Values"] > 0])


# Handle missing values by filling numerical columns with their median
df_cleaned = df.copy()
for col in df_cleaned.select_dtypes(include=[np.number]):  
    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Remove any remaining missing values
df_cleaned.dropna(inplace=True)

# Display results
st.write("All missing values have been handled. Below is the final count of missing values after processing:")

# Check and display missing values after processing
missing_after = df_cleaned.isnull().sum().sum()

# if missing_after > 0:
#     st.markdown(f'<p style="color:red; font-weight:bold;">Total missing values remaining: {missing_after}</p>', unsafe_allow_html=True)
# else:
#     st.markdown('<p style="color:green; font-weight:bold;">No missing values left in the dataset</p>', unsafe_allow_html=True)

st.write("Missing value count:", missing_after)






####################################### HANDLING UNWANTED COLUMNS #####################



st.subheader("Age Data Refinement")

age_counts = df_cleaned['Age'].value_counts().sort_index()
age_distribution = pd.DataFrame({
    'Age': age_counts.index,
    'Count': age_counts.values
})

chart = alt.Chart(age_distribution).mark_bar(color='#00A9AC').encode(
    x='Age:O',
    y='Count:Q'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart)

st.markdown(
    "As you can see in the age column, there are very few data points for ages 35 and above. These ages have to remove from the dataset to ensure that the analysis focuses on more common age groups."
)

df_cleaned = df_cleaned[df_cleaned['Age'] < 35]

st.write("Max Age in Dataset:", df_cleaned['Age'].max())






# # Count occurrences of each age
# age_distribution = df_cleaned["Age"].value_counts().sort_index()

# # Visualize Age distribution with a line chart
# st.subheader("Age Distribution")

# fig, ax = plt.subplots(figsize=(4, 2))
# sns.lineplot(x=age_distribution.index, y=age_distribution.values, marker="o", color="blue", ax=ax)
# ax.axvline(35, color="red", linestyle="dashed", linewidth=2, label="Threshold: 35")
# ax.set_title("Age Distribution (Before Filtering)")
# ax.set_xlabel("Age")
# ax.set_ylabel("Count")
# ax.legend()
# st.pyplot(fig)

# # Remove rows where Age is 35 or more
# df_cleaned = df_cleaned[df_cleaned["Age"] < 35]

# # Display message after removal
# st.write("Samples with Age ≥ 35 have been removed.")





# if 'Sleep Duration' in df_cleaned.columns:
#     st.subheader("🛏️ Sleep Duration Categories and Counts")
#     sleep_duration_counts = df_cleaned['Sleep Duration'].value_counts()
#     st.write(sleep_duration_counts)
# else:
#     st.write("The 'Sleep Duration' column does not exist in the dataset.")

# if 'Dietary Habits' in df_cleaned.columns:
#     st.subheader("🍽️ Dietary Habits Categories and Counts")
#     dietary_habits_counts = df_cleaned['Dietary Habits'].value_counts()
#     st.write(dietary_habits_counts)
# else:
#     st.write("The 'Dietary Habits' column does not exist in the dataset.")


# if 'Degree' in df_cleaned.columns:
#     st.subheader("🎓 Degree Categories and Row Counts")
#     degree_counts = df_cleaned['Degree'].value_counts()
#     st.write(degree_counts)
# else:
#     st.write("The 'Degree' column does not exist in the dataset.")


