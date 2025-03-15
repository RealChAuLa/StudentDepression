import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


from chapters.overview import df

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.image("./assets/preprocessingBanner.jpg", use_container_width=True)
st.title("Data Preprocessing")

st.write(
    "Data preprocessing is important because raw data is often messy, missing, or inconsistent. Without cleaning and preparing the data, models may give incorrect results. Preprocessing helps by fixing missing values, removing errors, and making the data easier for models to understand, which improves accuracy and performance."
)


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



####################################### HANDLING DUPLICATE ROWS #####################


st.subheader("Duplicate Rows Analysis")
duplicate_rows = df_cleaned.duplicated().sum()
st.write("Duplicate Rows count:", duplicate_rows)
if duplicate_rows > 0:
    st.write("Duplicate rows have been removed.")
    df_cleaned.drop_duplicates(inplace=True)
else:
    st.write("No duplicate rows found in the dataset.")


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





####################################### Sleep Duration Refinement #####################





st.subheader("Sleep Duration Refinement")

st.write("""
This section provides an overview of the user's sleep hours. The sleep durations are categorized into the following groups:
""")

sleep_counts = df_cleaned['Sleep Duration'].value_counts().reset_index()
sleep_counts.columns = ['Sleep Duration', 'Count']

# Sort by sleep duration (assuming it's numeric)
sleep_counts = sleep_counts.sort_values('Sleep Duration')

fig = px.bar(
    sleep_counts,
    x='Sleep Duration',
    y='Count',
    labels={'Sleep Duration': 'Hours of Sleep', 'Count': 'Number of Records'},
    text='Count',  # Display count on bars
    color='Sleep Duration',  # You can still keep color if needed
)

# Customize the chart
fig.update_traces(
    textposition='outside',
    texttemplate='%{text}',
    marker_line_width=1,
    marker_line_color='white',
    marker_color='#6ded66' #6ded66 # Apply the light green color to the bars
)

fig.update_layout(
    xaxis_title='Sleep Duration (hours)',
    yaxis_title='Count',
    coloraxis_showscale=False  # Hide the color scale
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Create a two-column layout
col1, col2 = st.columns(2)

# New paragraph about the "Others" category
st.write("""
It is worth noting that the "Others" category contains only a small amount of data. To maintain the focus on the primary sleep duration categories, the "Others" data will be excluded from the analysis.
""")

# Filter out the "Others" category
df_cleaned = df_cleaned[df_cleaned['Sleep Duration'] != 'Others']

# Display the updated count of  others category
others_count = df_cleaned[df_cleaned['Sleep Duration'] == 'Others'].shape[0]

st.write("Count of 'Others' category:", others_count)





####################################### Profession Refinement #####################




st.subheader("Profession Refinement")
profession_counts = df_cleaned['Profession'].value_counts().reset_index()
profession_counts.columns = ['Profession', 'Count']

# Sort by profession (if needed, assuming it's a categorical column)
profession_counts = profession_counts.sort_values('Profession')

fig = px.bar(
    profession_counts,
    x='Profession',
    y='Count',
    labels={'Profession': 'Profession', 'Count': 'Number of Records'},
    text='Count',  # Display count on bars
    color='Profession',  # Color by profession
)

# Customize the chart
fig.update_traces(
    textposition='outside',
    texttemplate='%{text}',
    marker_line_width=1,
    marker_line_color='white',
    marker_color='#b7ebb5'  # Apply the light green color to the bars
)

fig.update_layout(
    xaxis_title='Profession',
    yaxis_title='Count',
    coloraxis_showscale=False  # Hide the color scale
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write("""
In this analysis, only the "Student" profession will be considered, as the other professions contain only a small amount of data. Removing these less-represented professions helps focus the analysis on the more significant data points and ensures a clearer understanding of the distribution of the "Student" profession.
""")

#filter student profession
df_cleaned = df_cleaned[df_cleaned['Profession'] == 'Student']

st.write("Profession count after filtering:", df_cleaned['Profession'].value_counts().values[0])

# Drop the 'Profession' column as it's no longer needed
df_cleaned.drop(columns=['Profession'], inplace=True)

# Display message with emphasis
st.markdown("""
    <div style="background-color:#ffcccb; padding:10px; border-radius:5px;">
        <b style="color:#b30000;">The 'Profession' column has been removed
        since only 'Student' remains as the profession.</b> 
    </div>
""", unsafe_allow_html=True)





####################################### City Refinement #####################


#remove column city category  name 3



st.subheader("City Refinement")


# Get city counts and sort them
city_counts = df_cleaned['City'].value_counts().sort_index()
city_distribution = pd.DataFrame({
    'City': city_counts.index,
    'Count': city_counts.values
})

# Create bar chart for city distribution
chart = alt.Chart(city_distribution).mark_bar(color='#EA7369').encode(
    x='City:O',
    y='Count:Q'
).properties(
    width=600,
    height=400
)

# Display the chart in Streamlit
st.altair_chart(chart)


# remove city columns below 10
city_counts = df_cleaned['City'].value_counts()
city_counts = city_counts[city_counts > 10]
city_list = city_counts.index.tolist()

st.write("City count :", df_cleaned['City'].nunique())

st.write("""
To maintain data relevance and avoid skewed analysis, cities with fewer than 10 data entries have been removed. This ensures that only cities with a significant number of records are considered in the analysis.
""")
df_cleaned = df_cleaned[df_cleaned['City'].isin(city_list)]
st.write("City count :", df_cleaned['City'].nunique())






####################################### Dietary Refinement #####################


st.subheader("Dietary Refinement")

if 'Dietary Habits' in df_cleaned.columns:
    dietary_habits_counts = df_cleaned['Dietary Habits'].value_counts()
    st.write(dietary_habits_counts)

    # Explanation paragraph
    st.write("""
    In the 'Dietary Habits' column, the 'Others' category contains only a small amount of data. 
    To maintain data consistency and focus on the major dietary habits, we have removed entries categorized as 'Others.'
    """)

    # Remove 'Others' category
    df_cleaned = df_cleaned[df_cleaned['Dietary Habits'] != 'Others']

else:
    st.write("The 'Dietary Habits' column does not exist in the dataset.")


#count of others in column dietary habits

others_count = df_cleaned[df_cleaned['Dietary Habits'] == 'Others'].shape[0]
st.write("Count of 'Others' category:", others_count)





# ??????????????????????????????????????????????///////////////////////

df_cleaned = df_cleaned[df_cleaned['Degree'] != 'Others']







######################### handling categorical data ############################################################################


#gender encoding

st.subheader("Encoding")


st.write(df_cleaned[['Gender']].head(2))

st.write("""
To standardize the **Gender** column, we encode it as follows:
- **Male → 0**
- **Female → 1**

This conversion ensures that the gender data is in numerical format, making it easier for analysis and modeling.
""")

# Encode Gender column (Male → 0, Female → 1)
df_cleaned['Gender'] = df_cleaned['Gender'].map({'Male': 0, 'Female': 1})




st.write(df_cleaned[['Sleep Duration']].head())
# Explanation
st.write("""
To standardize the **Sleep Duration** column, we encode it as follows:
- **Less than 5 hours → 0**
- **5-6 hours → 1**
- **7-8 hours → 2**
- **More than 8 hours → 3**

This conversion ensures that sleep duration is represented numerically for better analysis.
""")

# Encode Sleep Duration column
sleep_duration_mapping = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
}

df_cleaned['Sleep Duration'] = df_cleaned['Sleep Duration'].map(sleep_duration_mapping)



st.write(df_cleaned[['Have you ever had suicidal thoughts ?']].head(2))

# Explanation
st.write("""
To standardize the **'Have you ever had suicidal thoughts ?'** column, we encode it as follows:
- **No → 0**
- **Yes → 1**

This conversion ensures that the data is in numerical format for better analysis.
""")

# Encode the column
df_cleaned['Have you ever had suicidal thoughts ?'] = df_cleaned['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})

# Display the updated dataframe

st.write(df_cleaned[['Family History of Mental Illness']].head(2))
st.write("""
To standardize the **'Family History of Mental Illness'** column, we encode it as follows:
- **No → 0**
- **Yes → 1**

This conversion ensures that the data is in numerical format for better analysis.
""")


# ????????????????????????????????????????///////////////////////city encode?????????????????///



st.subheader("Encoding")

# Explanation
st.write("""
To standardize the **City** column, we assign a unique numeric value to each city using Label Encoding.
This allows us to handle categorical data efficiently.
""")

# Apply Label Encoding
label_encoder = LabelEncoder()
df_cleaned['City'] = label_encoder.fit_transform(df_cleaned['City'])

# Display the encoded City column
# st.write("Encoded Data Sample:")
st.write(df_cleaned[['City']].head())


# Encode the column
df_cleaned['Family History of Mental Illness'] = df_cleaned['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})

# Display the updated dataframe
st.write("Encoded city:")


st.subheader("Encoding")

# Explanation
st.write("""
To standardize the **Dietary Habits** column, we assign a unique numeric value to each category using Label Encoding.
""")

# Apply Label Encoding
label_encoder = LabelEncoder()
df_cleaned['Dietary Habits'] = label_encoder.fit_transform(df_cleaned['Dietary Habits'])

# Display the encoded column
st.write("Encoded Dietary Habits")
st.write(df_cleaned[['Dietary Habits']].head())



st.subheader("Encoding - Degree")

# Explanation
st.write("""
To standardize the **Degree** column, we assign a unique numeric value to each category using Label Encoding.
""")

# Apply Label Encoding
label_encoder = LabelEncoder()
df_cleaned['Degree'] = label_encoder.fit_transform(df_cleaned['Degree'])

# Display the encoded column
st.write("Encoded Data Sample:")
st.write(df_cleaned[['Degree']].head())


# Remove the ID column if it exists
if 'id' in df_cleaned.columns:
    df_cleaned.drop(columns=['id'], inplace=True)



# Display the first 10 rows
st.write(df_cleaned.head(10))




