import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np

# Load the trained model
model = joblib.load('C:/Users/R.Nirmalraj/Desktop/New folder (3)/lgbm_model.pkl')

# Function to map categorical features back to their original values
def map_back(column, mapping):
    if isinstance(mapping, dict):
        reverse_mapping = {v: k for k, v in mapping.items()}
        return column.map(reverse_mapping)
    else:
        return column

# Streamlit app
# Page title
st.markdown("""
<div style="text-align: center; background-color: #f0f8ff; padding: 15px; border-radius: 10px;">
    <h1 style="color: #0073e6;">Prediction Model Results</h1>
</div>
""", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")

def main():
    # Read the dataset
    dataset = pd.read_csv('https://raw.githubusercontent.com/GuviMentor88/Training-Datasets/main/insurance_dataset.csv')
    dataset = dataset.dropna(axis=0)

    # Map categorical features to numeric values
    mapping = {
        'y': {'no': 0, 'yes': 1},
        'mon': {'jan': 2, 'feb': 6, 'mar': 11, 'apr': 7, 'may': 0, 'jun': 4, 'jul': 1, 'aug': 5, 'sep': 9, 'oct': 8, 'nov': 3, 'dec': 10},
        'education_qual': {'tertiary':3, 'secondary':1, 'unknown':2, 'primary':0},
        'marital': {'married':0, 'single':2, 'divorced':1},
        'call_type': {'unknown':0, 'cellular':2, 'telephone':1},
        'prev_outcome': {'unknown':0, 'failure':1, 'other':2, 'success':3},
        'job':{
            'management': 8,
            'technician': 4,
            'entrepreneur': 1,
            'blue-collar': 0,
            'unknown': 5,
            'retired': 10,
            'admin.': 7,  
            'services': 3,
            'self-employed': 6,
            'unemployed': 9,
            'housemaid': 2,
            'student': 11
        }
    }
    dataset.replace(mapping, inplace=True)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        job = st.selectbox('Job', dataset['job'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['job'])[0])
    with col2:
        marital = st.selectbox('Marital Status', dataset['marital'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['marital'])[0])
    with col3:
        education_qual = st.selectbox('Education Qualification', dataset['education_qual'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['education_qual'])[0])

    col4, col5, col6 = st.columns(3)
    with col4:
        call_type = st.selectbox('Call Type', dataset['call_type'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['call_type'])[0])
    with col5:
        prev_outcome = st.selectbox('Previous Outcome', dataset['prev_outcome'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['prev_outcome'])[0])
    with col6:
        mon = st.selectbox('Month', dataset['mon'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['mon'])[0])

    age = st.slider('Age', min_value=18, max_value=95, value=18)
    day = st.slider('Day', min_value=1, max_value=31, value=1)
    dur = st.slider('Duration (Seconds)', min_value=0, max_value=4918, value=0)
    num_calls = st.slider('Number of Calls', min_value=0, max_value=63, value=0)

    # Add a "Predict" button
    if st.button('Predict', key='predict_button'):
        # Prepare the input data for prediction
        data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education_qual': [education_qual],
            'call_type': [call_type],
            'day': [day],
            'mon': [mon],
            'dur': [dur], 
            'num_calls': [num_calls],
            'prev_outcome': [prev_outcome]
        })

        # Make prediction using the loaded model
        prediction = model.predict(data)[0]

        # Display the prediction
        if prediction == 0:
            st.error("❌ No, The customer is highly unlikely to subscribe to the insurance.")
        else:
            st.success("✅ Yes, The customer is highly likely to subscribe to the insurance.")

# Run the app
if __name__ == '__main__':
    main()
