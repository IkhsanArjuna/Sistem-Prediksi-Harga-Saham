import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM

# Konstanta
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# Judul aplikasi
st.title("Sistem Prediksi Harga Saham Bank - LSTM (7 Hari Prediksi)")

# # Pilihan saham
stocks = ('BBRI.JK', 'BBNI.JK', 'BBCA.JK' )




selected_stock = st.selectbox('Pilih Saham Untuk Diprediksi', stocks)
index = 0
for i in stocks:
    if selected_stock == i:
        break
    index+=1


    

# Prediksi selalu untuk 7 hari
period = 7

# Fungsi memuat data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Status loading
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Menampilkan data mentah
st.subheader('Raw data')
st.write(data.tail())

# Fungsi untuk memplot data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'][stocks[index]], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'][stocks[index]], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# Plot data mentah
plot_raw_data()

# Preprocessing data untuk LSTM
df = data[['Date', 'Close']].copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Membuat data untuk training
sequence_length = 60
X_train, y_train = [], []

for i in range(sequence_length, len(scaled_data)):
    X_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Membuat model LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=5)

# Membuat prediksi masa depan
test_data = scaled_data[-sequence_length:]
X_test = [test_data]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

future_predictions = []
for _ in range(period):
    prediction = model.predict(X_test)[0, 0]
    future_predictions.append(prediction)
    # Update test data untuk langkah berikutnya
    next_sequence = np.append(X_test[0, 1:], [[prediction]], axis=0)
    X_test = np.array([next_sequence])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Denormalisasi prediksi
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Menambahkan prediksi ke dataframe
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=period)
forecast_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Prediction'])

# Menampilkan hasil prediksi
st.subheader('Data perkiraan untuk 7 hari ke depan')
st.write(forecast_df)

# Plot hasil prediksi
st.write('Grafik perkiraan untuk 7 hari ke depan')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'][stocks[index]], name="Historical"))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Prediction'], name="Prediction"))
fig.layout.update(title_text='LSTM Prediksi (7 Hari Ke Depan)', xaxis_rangeslider_visible=False)
st.plotly_chart(fig)
