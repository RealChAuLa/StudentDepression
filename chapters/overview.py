import streamlit as st
import pandas as pd

# Load Dataset (Replace with actual dataset path)
@st.cache_data
def load_data():
    df = pd.read_csv("./data/Student_Depression_Dataset.csv")  # Update with actual file path
    return df

df = load_data()

st.image("./assets/overview_banner.png", use_container_width=True)

# Introduction to the Project
st.title("Introduction")
st.write(
    "In today's fast-paced academic environment, student mental health has become a growing concern. "
    "Depression among students can negatively impact their academic performance, social life, and overall well-being. "
    "By leveraging machine learning, we can develop predictive models to identify students at risk of depression early, "
    "allowing for timely intervention and support."
)

st.header("Why Predicting Student Depression is Important?")
st.write(
    "Student depression datasets help researchers, educators, and mental health professionals understand the key "
    "factors influencing student mental health. Early detection of depression can help in designing targeted interventions, "
    "providing better support systems, and ultimately improving student success rates."
)

st.markdown(
    "- **Early Identification**: Machine learning can help detect patterns that indicate potential depression risks.\n"
    "- **Improved Academic Performance**: Identifying at-risk students allows for timely academic and mental health support.\n"
    "- **Better Student Well-being**: Early interventions can improve overall student happiness and productivity.\n"
    "- **Data-Driven Decision Making**: Institutions can leverage insights from data to create better mental health policies."
)

st.header("Dataset Summary")
st.write(
    "The dataset used for this project focuses on analyzing, understanding, and predicting depression levels among students. "
    "It includes demographic information, academic performance, and lifestyle habits to identify patterns that contribute to student depression."
)

st.subheader("Key Features in the Dataset")
st.markdown(
    "- **ID**: Unique identifier for each student.\n"
    "- **Age**: Age of the student.\n"
    "- **Gender**: Student's gender (Male/Female).\n"
    "- **City**: Geographic location of the student.\n
    "- **CGPA**: Academic performance indicator (Grade Point Average).\n"
    "- **Sleep Duration**: Average daily sleep duration in hours.\n"
    "- **Academic Pressure**: Scale (0-5) indicating academic workload stress.\n"
    "- **Study Satisfaction**: Measure of how satisfied the student is with their studies.\n"
    "- **Dietary Habits**: Eating patterns that may impact mental health.\n"
    "- **Depression Status (Target Variable)**: Whether a student is experiencing depression (Yes/No)."
)

st.subheader("Dataset Overview")
st.info("The dataset consists of multiple records capturing student lifestyle and academic factors to analyze depression trends.")
st.write(f"Total Records: **{df.shape[0]}** |" f" Total Features: **{df.shape[1]}**")
st.dataframe(df.head())

st.header("Next Steps")
st.write(
    "In the upcoming sections, we will explore the dataset in detail, preprocess the data, engineer relevant features, and "
    "train machine learning models to predict student depression. The goal is to create an interactive and insightful analysis "
    "that helps in understanding and mitigating depression among students."
)
