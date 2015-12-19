import numpy
from numpy import matlib

from . import IClassifier
from utilities import *

class RBF():
    def __init__(self, n_X, n_RBF, n_y):
        self._n_X = n_X
        self._n_RBF = n_RBF
        self._n_y = n_y
        self._Theta1 = matlib.randn((n_RBF, n_X))
        self._Theta2 = matlib.randn((n_RBF + 1, n_y))
    
    def fit(self, X, y):
        # here will train the model using X and y
        pass
    
    def predict(self, X):
        # here will give the prediction of X
        # X is supposed to be an matrix or 2-d array and each sample shall be shaped in a row vector
        m = X.shape[0]
        RBF_result = matlib.empty((m, self._n_RBF + 1))
        
        for x_index in range(m):
            x_output = matlib.empty((1, self._n_RBF + 1))
            
            # RBF process
            for RBF_index in range(self._n_RBF):
                x_output[0, RBF_index + 1] = Euler_distance(X[x_index], self._Theta1[RBF_index])
            
            sigmoid_output = sigmoid(x_output)
            sigmoid_output[0] = 1
            
            RBF_result[x_index] = x_output
        
        # Logistic Regression Process
        raw_output = RBF_result * self._Theta2
        return sigmoid(raw_output)
    
    def test(self, X, y):
        # here will test with X and y
        # y is supposed to be of the same size of X in rows
        # now return the number of wrong cases
        y_predict = self.predict(X)
        return numpy.count_nonzero(numpy.sum((y_predict > 0.5) != y, axis=1))

IClassifier.register(RBF)