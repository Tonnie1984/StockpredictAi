import json
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
from utils import calculate_adx, calculate_vwma, calculate_stochastic


class StockPredictor:
    def __init__(self, symbol, api_key):
        self.symbol = symbol
        self.api_key = api_key
        self.data = None
        self.scaler = None
        self.date = datetime.now().strftime("%Y-%m-%d")

    def get_stock_data(self, interval='daily', outputsize='full'):
        base_url = 'https://www.alphavantage.co/query?'
        function = 'TIME_SERIES_DAILY'
        json_file = f'json/{self.symbol}_data.json'

        if os.path.exists(json_file) and self.date == datetime.fromtimestamp(
                os.path.getatime(json_file)).strftime(
                "%Y-%m-%d"):
            with open(json_file, 'r') as json_file:
                data = pd.read_json(json_file)
        else:
            url = f'{base_url}function={function}&symbol={self.symbol}&outputsize={outputsize}&apikey={self.api_key}'
            response = requests.get(url)
            data = response.json()
            with open(json_file, 'w') as json_file:
                json.dump(data, json_file)

        if 'Time Series (Daily)' in data:
            timeseries = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(timeseries, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            self.data = df


        else:
            raise ValueError('Limite de solicitudes Api alcanzado')

    def prepare_data(self):
        self.data['EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['EMA_200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        self.data['VWMA_50'] = calculate_vwma(self.data, 50)
        self.data['VWMA_200'] = calculate_vwma(self.data, 200)
        self.data['ADX'] = calculate_adx(self.data)
        self.data['Stochastic'] = calculate_stochastic(self.data)
        self.data.dropna(inplace=True)
        self.data['Price_Mean'] = (self.data['High'] + self.data['Low']) / 2

    def split_data(self):
        X = self.data[['Open', 'High', 'Low', 'Price_Mean', 'EMA_50', 'EMA_200', 'VWMA_50', 'VWMA_200', 'ADX',
                       'Stochastic']].astype(float)
        y = self.data['Close'].shift(-1).dropna()
        X = X[:-1]

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(1, 10)))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train_model(self, model, X_train, y_train, X_test, y_test):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stopping, reduce_lr])

    def predict_next_day(self, model, X_scaled):
        last_data_point = np.reshape(X_scaled[-1], (1, 1, X_scaled.shape[1]))
        next_day_prediction = model.predict(last_data_point)[0][0]
        last_close_price = self.data['Close'].iloc[-1]

        percentage_change = ((next_day_prediction - last_close_price) / last_close_price) * 100

        self.save_prediction(next_day_prediction, percentage_change, self.symbol)
        return next_day_prediction, percentage_change

    def save_prediction(self, prediction, percentage_change, symbol):
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        prediction_data = pd.DataFrame({
            'Fecha': [prediction_date],
            'Predicción': [prediction],
            'Variación Porcentual': [percentage_change],
            'Ticker': [symbol]
        })
        prediction_data.to_csv('saved_predictions/prediccion.csv', index=False, mode='a',
                               header=not os.path.exists('saved_predictions/prediccion.csv'))

    def get_scaler(self):
        return self.scaler
