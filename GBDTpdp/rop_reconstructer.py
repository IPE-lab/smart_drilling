import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import matplotlib as mpl
from sklearn.inspection import partial_dependence
mpl.rcParams['axes.unicode_minus'] = False  
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 

warnings.filterwarnings("ignore")

class GBDTModel:
    
    def __init__(self, path_train, path_test, pathtosave):
        # Initializes an instance of the GBDTModel class, specifying the file paths for the training data, test data, and the directory where the prediction results will be saved.
        self.path_train = path_train
        self.path_test = path_test
        self.pathtosave = pathtosave
        
    def read_data(self, path):
        # Reads data from an Excel file at the specified path, using the first row as the column names. Returns the data as a Pandas DataFrame.
        data = pd.read_excel(path, header=0)
        return data
    
    def train(self):
        #  Trains the gradient boosting regression model using the training data. The model is defined with hyperparameters such as the loss function, learning rate, and number of estimators, among others. The trained model is stored as an instance variable in the GBDTModel object.
        data_train = self.read_data(self.path_train)

        x_col = ['DEPTH', 'TQA', 'RPMA', 'HKLA', 'WOBA', 'TVA', 'MFOP', 'MFIA', 'MDIA', 'SPPA', 'BDTI', 'DMEA', 'DVER', 'MDOA']
        y_col = ['ROPA']

        x_train = data_train.loc[:, x_col]
        self.x_train = x_train
        y_train = data_train.loc[:, y_col]

        self.model_GBDT = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=300, 
                                                    max_depth=5, random_state=50, alpha=0.9)
        self.model_GBDT.fit(x_train, y_train)
    
    def predict(self):
        # Uses the trained gradient boosting regression model to predict the values of the response variable for the test data. The predicted values and the actual values are returned as NumPy arrays, and the predicted values are also written to an Excel file saved in the directory specified in the __init__() method.
        data_test = self.read_data(self.path_test)
        x_col = ['DEPTH', 'TQA', 'RPMA', 'HKLA', 'WOBA', 'TVA', 'MFOP', 'MFIA', 'MDIA', 'SPPA', 'BDTI', 'DMEA', 'DVER', 'MDOA']
        y_col = ['ROPA']
        x_test = data_test.loc[:, x_col]
        y_test = data_test.loc[:, y_col]

        y_test_predict = self.model_GBDT.predict(x_test)
        self.y_test = y_test
        self.y_test_predict = y_test_predict

        data_test.insert(data_test.shape[1], 'predication', y_test_predict)
        data_test.to_excel(self.pathtosave + '\predication.xlsx')

        return y_test, y_test_predict
    
    def predict_plot(self):
        # Creates a scatter plot of predicted values versus actual values from the test data, as well as a red line that represents perfect predictions. The resulting plot is saved as an image file in the directory specified in the __init__() method.
        inv_y, inv_yhat = list(np.array(self.y_test)), self.y_test_predict
        plt.scatter(inv_y, inv_yhat, color='black', linewidth=0.1)
        plt.plot(np.linspace(min(min(inv_y), min(inv_yhat)), max(max(inv_y), max(inv_yhat)), 2),
                    np.linspace(min(min(inv_y), min(inv_yhat)), max(max(inv_y), max(inv_yhat)), 2), color='red', linewidth=2.0)
        plt.xlabel('record(m/h)', fontsize=18)
        plt.ylabel('predication(m/h)', fontsize=18)
        plt.savefig(self.pathtosave + '\predication.png', bbox_inches='tight')

    def plot_pdp(self, feature_name):
        # Creates a 3D surface plot of the partial dependence of the response variable on two specified features. Returns the x, y, and z values used to construct the plot. The resulting plot is saved as an image file in the directory specified in the __init__() method.
        axes = partial_dependence(self.model_GBDT, self.x_train, feature_name)
        xx, yy = np.meshgrid(axes['values'][0], axes['values'][1])
        zz = axes['average'].reshape(xx.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, zz, cmap=plt.cm.RdBu, edgecolor='k')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='z', labelsize=22)
        ax.set_xlabel(feature_name[0], fontsize=30, labelpad=30)
        ax.set_ylabel(feature_name[1], fontsize=30, labelpad=30)
        ax.set_zlabel('pdp_value', fontsize=30, labelpad=25)

        ax.view_init(elev=25, azim=225)
        cax = fig.add_axes([0.78,0.2,0.03,0.6])
        cb = plt.colorbar(surf, cax=cax)
        cb.ax.tick_params(labelsize=20)
        plt.subplots_adjust(top=0.9)
        plt.savefig(self.pathtosave + '\%s-%s.png'%(feature_name[0], feature_name[1]), bbox_inches='tight')

        return xx, yy, zz