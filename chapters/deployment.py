import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set Streamlit page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Display header image
st.image("https://raw.githubusercontent.com/RealChAuLa/StudentDepression/master/assets/deployment_banner.jpg", use_container_width=True)

# Page Title
st.title("Depression Prediction Tool")

# Load the trained model and scaler
try:
    model = joblib.load('models/best_logistic_regression_model.pkl')  # Update with your best model's filename
    scaler = joblib.load('models/scaler.pkl')
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Reference dictionaries for encoding categorical variables
gender_dict = {'Male': 0, 'Female': 1}

city_dict = {
    'Agra': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chennai': 4,
    'Delhi': 5, 'Faridabad': 6, 'Ghaziabad': 7, 'Hyderabad': 8, 'Indore': 9,
    'Jaipur': 10, 'Kalyan': 11, 'Kanpur': 12, 'Kolkata': 13, 'Lucknow': 14,
    'Ludhiana': 15, 'Meerut': 16, 'Mumbai': 17, 'Nagpur': 18, 'Nashik': 19,
    'Patna': 20, 'Pune': 21, 'Rajkot': 22, 'Srinagar': 23, 'Surat': 24,
    'Thane': 25, 'Vadodara': 26, 'Varanasi': 27, 'Vasai-Virar': 28, 'Visakhapatnam': 29
}

sleep_duration_dict = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
}

dietary_habits_dict = {
    'Healthy': 0,
    'Moderate': 1,
    'Unhealthy': 2
}

degree_dict = {
    'B.Arch': 0, 'B.Com': 1, 'B.Ed': 2, 'B.Pharm': 3, 'B.Tech': 4,
    'BA': 5, 'BBA': 6, 'BCA': 7, 'BE': 8, 'BHM': 9, 'BSc': 10,
    'Class 12': 11, 'LLB': 12, 'LLM': 13, 'M.Com': 14, 'M.Ed': 15,
    'M.Pharm': 16, 'M.Tech': 17, 'MA': 18, 'MBA': 19, 'MBBS': 20,
    'MCA': 21, 'MD': 22, 'ME': 23, 'MHM': 24, 'MSc': 25, 'PhD': 26
}

yes_no_dict = {'No': 0, 'Yes': 1}

st.write("""
### Student Depression Assessment
Fill in the form below to get a prediction on depression risk based on our machine learning model.
""")

# Create form with three columns for better UI
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", options=list(gender_dict.keys()))
    age = st.number_input("Age", min_value=18, max_value=35, value=24)
    city = st.selectbox("City", options=list(city_dict.keys()))
    academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 3)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)

with col2:
    study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)
    sleep_duration = st.selectbox("Sleep Duration", options=list(sleep_duration_dict.keys()))
    dietary_habits = st.selectbox("Dietary Habits", options=list(dietary_habits_dict.keys()))
    degree = st.selectbox("Degree", options=list(degree_dict.keys()))

with col3:
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", options=list(yes_no_dict.keys()))
    work_study_hours = st.number_input("Daily Work/Study Hours", min_value=0, max_value=16, value=6)
    financial_stress = st.slider("Financial Stress Level (0-5)", 0, 5, 2)
    family_history = st.selectbox("Family History of Mental Illness", options=list(yes_no_dict.keys()))

# Create a prediction button
predict_button = st.button("Predict Depression Risk")

# When the button is clicked
if predict_button:
    # Collect all inputs into a dictionary
    input_data = {
        'Gender': gender_dict[gender],
        'Age': age,
        'City': city_dict[city],
        'Academic Pressure': academic_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Sleep Duration': sleep_duration_dict[sleep_duration],
        'Dietary Habits': dietary_habits_dict[dietary_habits],
        'Degree': degree_dict[degree],
        'Have you ever had suicidal thoughts ?': yes_no_dict[suicidal_thoughts],
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': yes_no_dict[family_history]
    }

    # Convert to DataFrame (required for scaling with the same format used during training)
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Display results with formatting
    st.markdown("---")
    st.subheader("Prediction Results")

    # Create a colored box based on the prediction
    if prediction[0] == 1:
        depression_risk = "High"
        risk_color = "#f63366"  # Red color for high risk
    else:
        depression_risk = "Low"
        risk_color = "#0068c9"  # Blue color for low risk

    # Calculate the probability percentage
    risk_percentage = probability[0][1] * 100

    # Display the result with styling
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}; color: white;">
            <h3 style="text-align: center; margin: 0;">Depression Risk: {depression_risk}</h3>
            <p style="text-align: center; font-size: 18px; margin: 10px 0;">
                Probability: {risk_percentage:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display additional information based on risk level
    st.markdown("### Assessment Details")

    if prediction[0] == 1:
        st.markdown("""
        #### High Depression Risk Factors:
        - **Academic Factors**: High academic pressure combined with low satisfaction can contribute to stress.
        - **Sleep Pattern**: Inadequate sleep duration is strongly associated with depression risk.
        - **Personal History**: Previous suicidal thoughts are a significant risk factor.

        #### Recommendations:
        1. **Seek Professional Help**: Consider consulting with a mental health professional
        2. **Academic Support**: Speak with academic advisors about managing workload
        3. **Improve Sleep Habits**: Work on establishing better sleep patterns
        4. **Build Support Network**: Connect with friends, family and support groups
        """)
    else:
        st.markdown("""
        #### Low Depression Risk Assessment:
        - Your responses indicate a lower risk profile for depression
        - Continue maintaining healthy habits and self-care practices

        #### Recommendations:
        1. **Maintain Balance**: Continue balancing academic demands with personal time
        2. **Regular Check-ins**: Periodically assess your mental wellbeing
        3. **Preventive Care**: Practice stress management techniques
        4. **Healthy Lifestyle**: Continue with good sleep, diet and exercise habits
        """)

    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer**: This tool provides an estimate based on statistical patterns and should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing mental health concerns, please consult with a qualified healthcare provider.
    """)

else:
    # Display information about the tool when not yet submitted
    st.markdown("""
    ### How this works:
    1. Fill in all the fields in the form above with your information
    2. Click the "Predict Depression Risk" button
    3. The model will analyze your inputs and provide a risk assessment
    4. This assessment is based on patterns identified from student data

    **Note**: All information entered is processed locally and not stored or shared.
    """)

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This prediction tool uses a machine learning model trained on student mental health data. 
    The model has been tuned to identify patterns associated with depression risk factors among students.

    Key features that influence the prediction include:
    - Academic factors (pressure, satisfaction, CGPA)
    - Lifestyle factors (sleep duration, dietary habits)
    - Personal history (previous suicidal thoughts, family history)
    - Demographic information (age, gender)

    The model has been trained on data from student populations and may not be as accurate for non-students or individuals outside the age range of 18-35.
    """)