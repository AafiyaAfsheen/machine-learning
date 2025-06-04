
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HUAWEI\machine-learning\loan-train.csv")
    return df

def preprocess_data(df):
    df = df.copy()
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    df.drop('Loan_ID', axis=1, inplace=True)

    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def train_and_save_model(df):
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    with open("loan_model.pkl", "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return acc, cm, cr

def load_model():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

def main():
    st.title("Loan Approval Prediction")

    df = load_data()
    df_clean = preprocess_data(df)

    if not os.path.exists("loan_model.pkl"):
        acc, cm, cr = train_and_save_model(df_clean)
        st.write("Model trained successfully!")
        st.write(f"Accuracy: {acc:.2f}")
        st.text("Confusion Matrix:\n" + str(cm))
        st.text("Classification Report:\n" + str(cr))
    else:
        st.success("Model already trained.")

    model = load_model()

    st.header("Enter Applicant Information")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=12, value=360)
    credit_history = st.selectbox("Credit History", ['Good (1.0)', 'Bad (0.0)'])
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    input_data = pd.DataFrame({
        'Gender': [0 if gender == 'Female' else 1],
        'Married': [1 if married == 'Yes' else 0],
        'Dependents': [0 if dependents == '0' else 1 if dependents == '1' else 2 if dependents == '2' else 3],
        'Education': [0 if education == 'Graduate' else 1],
        'Self_Employed': [1 if self_employed == 'Yes' else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [1.0 if credit_history.startswith('Good') else 0.0],
        'Property_Area': [2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        result = "Approved ✅" if prediction == 1 else "Rejected ❌"
        st.subheader(f"Loan Application Status: {result}")

if __name__ == '__main__':
    main()
