import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Helper function to clean binary columns
def clean_binary_column(column, positive_value):
    return np.where(column == positive_value, 1, 0)

# Streamlit app title
st.title("LinkedIn Usage Prediction")

# File upload section
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif os.path.exists("social_media_usage.csv"):
    data = pd.read_csv("social_media_usage.csv")
    st.success("Data loaded successfully!")
else:
    st.error("No data file provided. Please upload 'social_media_usage.csv'.")
    st.stop()  # Stops execution if no data is provided

# Ensure required columns exist
required_columns = {'web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age'}
if not required_columns.issubset(data.columns):
    st.error("The dataset is missing required columns. Please check the file.")
    st.stop()

# Data processing
data = data.copy()  # Avoid modifying original data
data.loc[:, 'sm_li'] = clean_binary_column(data['web1h'], 1)  # LinkedIn usage
data.loc[:, 'income'] = data['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)  # Valid income range
data.loc[:, 'education'] = data['educ2'].apply(lambda x: x if 1 <= x <= 8 else np.nan)  # Valid education range
data.loc[:, 'parent'] = clean_binary_column(data['par'], 1)  # Parent status
data.loc[:, 'married'] = clean_binary_column(data['marital'], 1)  # Marital status
data.loc[:, 'female'] = clean_binary_column(data['gender'], 2)  # Female gender
data.loc[:, 'age'] = data['age'].apply(lambda x: x if 0 <= x < 98 else np.nan)  # Valid age range

# Filter dataset and drop rows with missing values
features = ['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']
processed_data = data[features].dropna()

# Check if enough data is available for training
if processed_data.shape[0] < 10:
    st.error("Not enough data available after cleaning. Please check the dataset.")
    st.stop()

# Separate features (X) and target (y)
X = processed_data[['income', 'education', 'parent', 'married', 'female', 'age']]
y = processed_data['sm_li']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

st.subheader("Prediction Section")

# Inputs for new individuals
st.write("Enter details of individuals to predict LinkedIn usage probability:")
income = st.number_input("Income (1-9)", min_value=1, max_value=9, step=1, value=8)
education = st.number_input("Education (1-8)", min_value=1, max_value=8, step=1, value=7)
parent = st.selectbox("Is a parent?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
married = st.selectbox("Is married?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
female = st.selectbox("Is female?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
age = st.number_input("Age (valid range: 0-97)", min_value=0, max_value=97, step=1, value=42)

# Predict probabilities for LinkedIn usage
if st.button("Predict"):
    individual = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [parent],
        'married': [married],
        'female': [female],
        'age': [age]
    })
    try:
        probability = model.predict_proba(individual)[0, 1]  # Fix indexing error
        st.write(f"Probability of LinkedIn usage for age {age}: {probability:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
