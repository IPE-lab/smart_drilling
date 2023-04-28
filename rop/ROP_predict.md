## GBDTLSTMModel Documentation

The `GBDTLSTMModel` is a Python class used for time series prediction using the GBDT-LSTM model. The model combines shallow and deep learning techniques to effectively extract information from the data and handle non-linear relationships to improve prediction accuracy.

### Methods

- `__init__(self, path_train, path_predict, test_ratio=0.2, n_past=80, optimizer='adam')`: The constructor of the class. It initializes the GBDT-LSTM model and loads the training and prediction data from the specified paths. It takes the following parameters:
  - `path_train`: The path to the training data file.
  - `path_predict`: The path to the data file used for prediction.
  - `test_ratio`: The ratio of test data to total data used for model evaluation. Default is 0.2.
  - `n_past`: The number of time steps in the past used for predicting future values. Default is 80.
  - `optimizer`: The optimizer used for training the LSTM model. Default is 'adam'.
- `read_data(self, path)`: Reads data from the specified path and returns it as a Pandas DataFrame.
- `train_gbdt(self, data_train)`: Processes and trains the given training data to generate the GBDT model.
- `createXY(self, datasets)`: Converts the dataset to the input and output format suitable for the LSTM model.
- `_build_and_fit_model(self, X_train, y_train, X_test, y_test)`: Builds and fits the LSTM model.
- `train_lstm(self, data_train)`: Processes and trains the given training data to generate the LSTM model.
- `lstm_predict(self, X_test)`: Uses the trained LSTM model to predict the values of the test data and returns the predictions.
- `predict(self)`: Predicts the values of the test data using the GBDT-LSTM model. It returns the real values, GBDT model predictions, and LSTM model predictions. The real values and GBDT model predictions are used for training the LSTM model, while the LSTM model predictions are used for final prediction.

### Usage

To use the `GBDTLSTMModel` class for time series prediction, you need to create an object of the class and specify the paths to the training and prediction data files. Then, you can use the `predict` method to predict the values of the test data. 

```python
path_train = 'path_train.xlsx'
path_predict = 'path_predict.xlsx'
model = GBDTLSTMModel(path_train, path_predict)
y_test, y_test_predict, y_pred_future = model.predict()

print(y_test, y_test_predict, y_pred_future)
``` 

### Limitations

The `GBDTLSTMModel` class has some limitations that need to be addressed such as:

- It does not perform model selection or hyperparameter tuning, which may lead to underfitting or overfitting.
- The data processing workflow is hard-coded in the code and may not be suitable for other types of data, and therefore should be refactored.
- Error handling needs to be added to improve code stability.
- It does not provide methods for model evaluation, making it difficult to assess the model's performance and effectiveness.