import streamlit as st
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("ML Model Deployment with Streamlit")
st.write("Enter the input features to get predictions:")

feature1 = st.selectbox("Education [Graduate -> 0 & not Graduate -> 1] :",('0', '1'))
feature2 = st.selectbox("Self Employee [Yes -> 1 & No -> 0] :",('0', '1'))
feature3 = st.slider("Annual Income :", min_value=0, max_value=50000000, step=100)
feature4 = st.slider("Loan Amount :", min_value=0, max_value=50000000, step=500)
feature5 = st.slider("Loan Term :", min_value=0, max_value=100, step=1)
feature6 = st.slider("Cibil Score :", min_value=0, max_value=1200, step=1)
feature7 = st.slider("Residential Assets Value :", min_value=0, max_value=50000000, step=1000)
feature8 = st.slider("Commercial Assets Value :", min_value=0, max_value=50000000, step=250)
feature9 = st.slider("Luxury Assets Value :", min_value=0, max_value=50000000, step=500)
amount = st.slider("Bank Asset Value :", min_value=0, max_value=50000000, step=100)

if st.button("Predict"):
    input_features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, amount]])
    prediction = model.predict(input_features)
    if(prediction[0]==0):
        st.success("Prediction: Loan Approved")
    else:
        st.error("Prediction: Loan Rejected")
