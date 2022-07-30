import streamlit as st
import pandas as pd
import numpy as np
import pickle

df1 = pd.read_csv("telecom_data.csv")

choice = ["Yes", "No"]
choice1 = ["Yes", "No"]
multiple_lines = ["No phone service", "Yes", "No"]
gender = ["Male", "Female"]
internet_service = ["DSL", "No", "Fiber optic"]
contract = ["Month-to-month", "One year", "Two year"]
paymentMethod = ["Electronic check", "Mailed check",
    "Credit card (automatic)", "Bank transfer (automatic)"]
online = ["Yes", "No", "No internet service"]

model = pickle.load(open("model.pkl", "rb"))


header = st.container()
inputs = st.container()


with header:
    st.title("Telecom Churn Prediction")

with inputs:
    SeniorCitizen = st.selectbox("Is he Senior Citizen", sorted(choice))
    MonthlyCharges = st.text_input("Monthly Charges", "Type Here")
    TotalCharges = st.text_input("Total Charges", "Type Here")
    tenure = st.text_input("Total tenure", "Type Here")
    Gender = st.selectbox("Gender", sorted(gender))
    Partner = st.selectbox("Does he have a partner", sorted(choice))
    Dependents = st.selectbox("Does he have a Dependents", sorted(choice))
    phoneService = st.selectbox("Does he avail phone service", sorted(choice))
    multipleLines = st.selectbox(
        "Does he have multiple lines", sorted(multiple_lines))
    InternetService = st.selectbox(
        "Does he have internet service", sorted(internet_service))
    OnlineSecurity = st.selectbox(
        "Does he have Online Security", sorted(online))
    OnlineBackup = st.selectbox("Does he have Online Backup", sorted(online))
    DeviceProtection = st.selectbox(
        "Does he have Device Protection", sorted(online))
    TechSupport = st.selectbox("Does he have Tech Support", sorted(online))
    StreamingTV = st.selectbox("Does he have Streaming TV", sorted(online))
    StreamingMovies = st.selectbox(
        "Does he have Streaming Movies", sorted(online))
    Contract = st.selectbox(
        "Which type of Contract does he have", sorted(contract))
    PaperlessBilling = st.selectbox(
        "Does he payment on Paperless Bill", sorted(choice))
    PaymentMethod = st.selectbox("Payment Method", sorted(paymentMethod))

data = [[SeniorCitizen, MonthlyCharges, TotalCharges, tenure, Gender, Partner, Dependents, phoneService, multipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]]

input_df = pd.DataFrame(data, columns=[ 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])


df_2 = pd.concat([df1, input_df], ignore_index=True)
 # Group the tenure in bins of 12 months

df_2.dropna(inplace=True)
labels = [f"{i}-{i+12}" for i in range(0, 72, 12)]

df_2['tenure_grp'] = pd.cut(df_2.tenure.astype(
    int), range(0, 80, 12), right=False, labels=labels)

# drop column customerID and tenure
df_2.drop(columns=['tenure'], axis=1, inplace=True)


new_df__dummies = pd.get_dummies(df_2,columns=[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_grp']],drop_first=True)

# st.write(new_df__dummies)

# single = model.predict(new_df__dummies.tail(1))
# probablity = model.predict_proba(new_df__dummies.tail(1))[:, 1]

st.table(new_df__dummies)
# result = model.predict_proba(input_df)

# if result == 1:
#         o1 = "This customer is likely to be churned!!"
#         o2 = "Confidence: {}".format(result*100)
# else:
#         o1 = "This customer is likely to continue!!"
#         o2 = "Confidence: {}".format(result*100)


# st.header(o1)
# st.header(o2)