import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


### load the trained model

model = tf.keras.models.load_model('model.h5')


## load the encoders and scaler

with open('Label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encode_geo.pk1', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)


### streamlit app

st.title('Customer Churn Prediction')

## users input
geography = st.selectbox('Geography',  onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tensure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of product', 1, 4)
has_cr_card = st.selectbox('Has Credit card', [0,1])
is_active_number = st.selectbox('Is active Member', [0,1])



# prepare the input data.
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender' :[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tensure],
    'Balance':[balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_number],
    'EstimatedSalary':[estimated_salary]
})

## One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


## combine onhot with columns with inputdata

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


### Scale the input data
input_data_sacled = scaler.transform(input_data)

## prediction on Churn
prediction = model.predict(input_data_sacled)
prediction_proba = prediction[0][0]

st.write(f'Churu Probability: {prediction_proba: .2f}')

if prediction_proba > 0.5:
    st.write('The Customer is likely to churu.')
else:
    st.write('The Customer is not likely to churu.')