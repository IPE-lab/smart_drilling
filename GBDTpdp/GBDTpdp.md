## Introduction

The `GBDTModel` class is a Python implementation of a gradient boosting regression model that includes various methods for reading data, training the model, making predictions, and visualizing the results.

### Methods

- `__init__(self, path_train, path_test, pathtosave)`: Initializes an instance of the GBDTModel class.
- `read_data(self, path)`: Reads data from an Excel file at the specified path.
- `train(self)`: Trains the gradient boosting regression model using the training data.
- `predict(self)`: Uses the trained gradient boosting regression model to predict the values of the response variable for the test data.
- `predict_plot(self)`: Creates a scatter plot of predicted values versus actual values from the test data.
- `plot_pdp(self, feature_name)`: Creates a 3D surface plot of the partial dependence of the response variable on two specified features.

### Usage

To use the `GBDTLSTMModel` class for time series prediction, you need to create an object of the class and specify the paths to the training and prediction data files. Then, you can use the `predict` method to predict the values of the test data. 

```python
path_train = 'path_train.xlsx'
path_predict = 'path_predict.xlsx'
path_save = 'save'

first_feature = 'DEPTH'
second_feature = 'TQA'

model = GBDTModel(path_train , path_predict, path_save)
model.train()
model.predict()
model.plot_pdp(['DEPTH', 'TQA'])
``` 

### Limitations

The `GBDTModel` class has some limitations that need to be addressed such as:

- It does not perform model selection or hyperparameter tuning, which may lead to underfitting or overfitting.
- The data processing workflow is hard-coded in the code and may not be suitable for other types of data, and therefore should be refactored.
- Error handling needs to be added to improve code stability.