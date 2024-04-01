import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


model = pickle.load(open('knn_model.sav', 'rb'))

st.set_page_config(page_title='QuantitySales Prediction App', layout='wide')
st.title('QuantitySales Prediction App')
st.text("""This product is for predicting sales,so that businesses can allocate 
resources effectively, avoid stockouts or overstocking, optimize their 
operations, and enable businesses to make informed decisions about pricing, 
promotions, and discounts.""")
st.sidebar.header('QuantitySales Prediction Scale')


# function
def main_user():
    price = st.sidebar.slider('Price', 1000, 10000, 1)
    category = st.sidebar.slider('Category', 2, 10, 0)
    shopping_mall = st.sidebar.slider('Shopping Mall', 5, 10, 0)
    Year = st.sidebar.selectbox('Year', ('2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028'))

    mainuser_reprtdata = {
        'price': price,
        'Year': Year,
        'category': category,
        'shopping_mall': shopping_mall
    }
    user_report_data = pd.DataFrame(mainuser_reprtdata, index=[0])
    user_data = user_report_data.loc[:, ['price', 'Year', 'category', 'shopping_mall']]
    return user_report_data


user_data = main_user()
st.header('QuantitySales Prediction App')
st.write(user_data)

quantity = model.predict(user_data)
st.subheader('Prediction')
st.subheader(str(np.round(quantity[0], 2)))
