from pyexpat import model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#machine algorithms

from sklearn import svm   
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.decomposition import PCA 
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor


#from sklearn.svm import SVR
import azutils
import math
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import neighbors
from sklearn import metrics



param_C_linear = 10000
param_C_poly = 100
param_C_rbf = 1000
param_C_sigmoid = 1000 




def normalize_data(data_x, data_y):
    scalerx = MinMaxScaler().fit(data_x)
    X_std = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler() 
    Y_std = data_y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, 
                                            train_size = 0.80,random_state=0)

    return X_train, X_test, y_train, y_test

def aznormalize_data(data_x, data_y):
    scalerx = MinMaxScaler().fit(data_x)
    X_std = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler() 
    Y_std = data_y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, 
                                            train_size = 0.80,random_state=0)

    return X_train, X_test, y_train, y_test,X_std,Y_std





############ az support vector Machines linear ########################
def azsupport_vector_regresyon_linear(X_train, y_train, X_test,y_test,_page, plotting):

    #X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    svmmodel = svm.SVR(C=param_C_linear, kernel = 'linear') 
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    svmodel=svmmodel.fit(X_train, y_train)
   
    train_score = svmodel.score(X_train, y_train)
    test_score = svmodel.score(X_test, y_test)

    #print(svmmodel.coef_)
    #print(feature_names)


    #azutils.azplot_f_importance(svmmodel.coef_, feature_names,k,"linear_svr",_page)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score

