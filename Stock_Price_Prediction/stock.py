import numpy as np
import pandas as pd 
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



model = load_model('\Stock_Price_Prediction\Stock Prediction Model.keras')


today = date.today()
start = '2014-01-01'
end = today

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symnbol','RELIANCE.NS')

data = yf.download(stock,start,end)

# data_show = data.iloc[::-1].reset_index(drop=True)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])


Scalar = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test],ignore_index=True)
data_test_scale = Scalar.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig1)


st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'g')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig2)


st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,8))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'g')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig3)

x = []
y= []

for i in range (100 , data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])


x,y = np.array(x),np.array(y)

y_predict = model.predict(x)


scale = 1/Scalar.scale_

y_predict = y_predict*scale

y = y*scale



st.subheader('Orignal Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,8))
plt.plot(y,'r', label = 'Original Price')
plt.plot(y_predict,'b', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()  
plt.show()
st.pyplot(fig4)
