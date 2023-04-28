import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense

class GBDTLSTMModel:
    def __init__(self, path_train, path_predict, test_ratio=0.2, n_past=80, optimizer='adam'):
        self.path_train = path_train
        self.path_predict= path_predict
        self.test_ratio = test_ratio
        self.n_past = n_past
        self.optimizer = optimizer
        
        self.feature_cols = ['DEPTH', 'TQA', 'RPMA', 'HKLA', 'WOBA', 'TVA', 'MFOP', 'MFIA', 'MDIA', 'SPPA', 'BDTI', 'DMEA', 'DVER', 'MDOA']
        self.target_col = ['ROPA']

    def read_data(self, path):
        # Read data from a given path and return the data as a pandas DataFrame.
        data = pd.read_excel(path, header=0)
        return data

    def train_gbdt(self, data_train):
        # Train a GBDT model using the given training data.
        x_train = data_train.loc[:, self.feature_cols]
        y_train = data_train.loc[:, self.target_col]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)

        self.scaler_GBDT = scaler
        self.model_GBDT = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=300,
                                                    max_depth=5, random_state=50, alpha=0.9)
        self.model_GBDT.fit(x_train, y_train)

    def createXY(self, datasets):
        # Create input and output data for LSTM model.
        dataX = []
        dataY = []
        for i in range(self.n_past, len(datasets)):
            dataX.append(datasets[i - self.n_past:i, 0:datasets.shape[1]])
            dataY.append(datasets[i, 0])
        return np.array(dataX), np.array(dataY)

    def _build_and_fit_model(self, X_train, y_train, X_test, y_test):
        # Build and fit a LSTM model and return the model.
        grid_model = Sequential()
        grid_model.add(LSTM(100, return_sequences=True, input_shape=(self.n_past, self.n_features)))
        grid_model.add(LSTM(200))
        grid_model.add(Dense(1))
        grid_model.compile(loss='mse', optimizer=self.optimizer)

        grid_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        return grid_model

    def train_lstm(self, data_train):
        # Train a LSTM model using the given training data.
        x_train_scaled = self.scaler_GBDT.transform(data_train)
        df_training_scaled = pd.DataFrame(x_train_scaled, columns=self.feature_cols)
        df_training_scaled[self.target_col] = 0
        self.df = df_training_scaled
        X_train, y_train = self.createXY(df_training_scaled.values)
        self.model = self._build_and_fit_model(X_train, y_train, X_train[:100], y_train[:100])

    def lstm_predict(self, X_test):
        # Predict output values of LSTM model using the given test data and return the predicted values. 
        X_test_scaled = self.scaler_GBDT.transform(X_test)
        X_test_reframed = self.createXY(X_test_scaled)[0]
        y_pred = self.model.predict(X_test_reframed)
        y_pred_unscaled = y_pred.reshape((len(y_pred), 1))
        return y_pred_unscaled

    def predict(self):
        # Predict target values of test data using the trained GBDT and LSTM models.
        data_train = self.read_data(self.path_train)
        self.train_gbdt(data_train)
        
        data_test = self.read_data(self.path_predict)
        x_test = data_test.loc[:, self.feature_cols]
        y_test = data_test.loc[:, self.target_col]

        x_test_scaled = self.scaler_GBDT.transform(x_test)
        y_test_predict = self.model_GBDT.predict(x_test_scaled)

        df_past = pd.concat([x_test.head(1000), pd.DataFrame(y_test_predict[:1000], columns=self.target_col)], axis=1)

        self.train_lstm(df_past)

        y_pred_future = self.lstm_predict(x_test.iloc[1000:])

        return y_test, y_test_predict, y_pred_future