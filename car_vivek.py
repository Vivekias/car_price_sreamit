import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('bagging.pkl', 'rb'))

def main():
    st.title('Car Price Prediction Using ML')
    img = 'car.png'
    st.image(img)
    st.subheader('Car Price Predictor')

    df = pd.read_csv('CAR DETAILS.csv')
    cars = df['name'].unique()
    transmission = df['transmission'].unique()
    seller = df['seller_type'].unique()
    owner = df['owner'].unique()
    fuel = df['fuel'].unique()

    p1 = st.selectbox('Select the Car', cars)

    p2 = st.slider('Model Year', 2005, 2020, 2005)

    p3 = st.selectbox('Seller Type', seller)
    p3_mapping = {'Individual': 1, 'Dealer': 0, 'Trustmark Dealer': 2}
    p3 = p3_mapping[p3]

    p4 = st.selectbox('Owner Type', owner)
    p4_mapping = {
        'First Owner': 0,
        'Second Owner': 2,
        'Third Owner': 4,
        'Fourth & Above Owner': 1,
        'Test Drive Car': 3
    }
    p4 = p4_mapping[p4]

    p5 = st.selectbox('Transmission Type', transmission)
    p5_mapping = {'Manual': 1, 'Automatic': 0}
    p5 = p5_mapping[p5]

    p6 = st.selectbox('Fuel Type', fuel)
    p6_mapping = {'Petrol': 4, 'Diesel': 1, 'CNG': 0, 'LPG': 3, 'Electric': 2}
    p6 = p6_mapping[p6]

    p7 = (st.slider('KM Driven', 500, 10000000, 500)) / 100000

    x = pd.DataFrame({'year': [p2], 'fuel': [p6], 'seller_type': [p3],
                      'transmission': [p5], 'owner': [p4], 'km_driven_in_lacks': [p7]})

    ok = st.button('Predict Car Price')
    if ok:
        prediction = model.predict(x)
        st.success('Predicted Car Price: ' + str(prediction * 100000) + ' Rupees')
        st.caption('Thanks for using!')
        st.balloons()
        st.write('Created by Vivek Kumar Singh')

if __name__ == '__main__':
    main()