import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class LstmCnnModel:
    def __init__(self, path_train, path_predict, learning_rate=0.001, 
                 beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=1e-4, num_lstm_layers=2, 
                 lstm_hidden_size=16, cnn_num_units=128, num_epochs=5, validation_split_ratio=0.1, batch_size=128):
        # Initializes the LstmCnnModel class by setting various parameters and hyperparameters for the neural network model.
        self.path_train = path_train
        self.path_predict = path_predict

        self.X_predict = None

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_num_units = cnn_num_units
        self.num_epochs = num_epochs
        self.validation_split_ratio = validation_split_ratio
        self.batch_size = batch_size

        self.device = "cpu"
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self._load_data()
        self._build_model()

    def _load_data(self):
        # Loads the dataset specified by the file paths provided in init(), preprocesses it, and stores it in instance variables.
        self.data_train = pd.read_excel(self.path_train, header=0)
        self.data_predict = pd.read_excel(self.path_predict, header=0) 

        X_train = self.data_train.iloc[:, 1:9].values
        y_predict = self.data_train.iloc[:, 9].values

        self.X_predict = self.data_predict.iloc[:, 1:9].values

        self.X_train = self.scaler.fit_transform(X_train)
        self.y_predict = tf.keras.utils.to_categorical(y_predict, num_classes=4)
    
    def _build_model(self):
        # Constructs an LSTM-CNN hybrid model in Tensorflow 2.0.
        inputs = tf.keras.layers.Input(shape=(self.X_train.shape[1],))
        x = tf.keras.layers.Reshape((self.X_train.shape[1], 1))(inputs)

        x = tf.keras.layers.Conv1D(16, 128, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(16, 256, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(16, 128, padding='same', activation='relu')(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Reshape((1, -1))(x)
        for i in range(self.num_lstm_layers):
            x = tf.keras.layers.LSTM(self.lstm_hidden_size, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self.lstm_hidden_size, return_sequences=False)(x)

        x = tf.keras.layers.Concatenate()([x, tf.keras.layers.GlobalAveragePooling1D()(inputs)])

        x = tf.keras.layers.Dense(self.cnn_num_units, activation='relu')(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, 
                                             epsilon=self.epsilon, decay=self.weight_decay)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    def train(self):
        # Trains the LSTM-CNN model using the training data provided in init().
        self.model.fit(self.X_train, self.y_predict, batch_size=self.batch_size, 
                       epochs=self.num_epochs, validation_split=self.validation_split_ratio)

    def predict(self):
        # Predicts the labels of new data using the trained LSTM-CNN model.
        X_predict_input = np.expand_dims(self.X_predict, axis=2)

        y_pred_future = self.model.predict(X_predict_input)

        return y_pred_future
