import streamlit as st
import pandas as pd
import joblib

# Load the saved model and encoder
model = joblib.load('churn_model.pkl')
le = joblib.load('contract_encoder.pkl')

st.title("Customer Churn Prediction App")
st.write("Enter the customer's details below to predict if they are at risk of leaving.")

# Create input sliders and dropdowns for the user
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
total_charges = st.number_input("Total Charges ($)", value=600.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Prediction button
if st.button("Predict Churn Risk"):
    # Convert the string contract back to the number the model understands
    contract_encoded = le.transform([contract])[0]
    
    # FIX: Changed column names to be strictly lowercase to match the trained model
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract_encoded]], 
                              columns=['tenure', 'monthlycharges', 'totalcharges', 'contract'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk: This customer is likely to churn.")
    else:
        st.success("✅ Low Risk: This customer is likely to stay.")