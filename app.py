import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

st.title('Medical Diagnostic Web App ')

# Step1 : Load the model

model = open('rfc.pickle','rb')
clf = pickle.load(model)
model.close()

#Step 2: Get the frontend user input
pregs=st.number_input('Pregnancies',1,20, 1)

glucose = st.slider('Glucose',40.0, 200.0, 40.0)

bp = st.slider('BloodPressure',24.0, 122.0, 24.0)

skin = st.slider('SkinThickness',7.0 ,99.0 ,7.0 )

insulin =st.slider('Insulin',18.0,850.0,18.0)

bmi = st.slider('BMI',18.0,67.0,18.0)

dpf=st.slider('DiabetesPedigreeFunction',0.5,2.5,0.05)

age = st.slider('Age', 21,81,21)

#Step 3: converting user input to model input
data = {'Pregnancies':pregs, 'Glucose':glucose, 'BloodPressure':bp, 'SkinThickness':skin, 'Insulin':insulin,
       'BMI':bmi, 'DiabetesPedigreeFunction':dpf, 'Age':age}

input_data= pd.DataFrame([data])

# Step 4: Get the predictions

preds = clf.predict(input_data)[0]
if st.button('Predict'):
    if preds==1:
        st.error('The person has Diabetes')
    if preds==0:
        st.success('The person Diabetes free')
