import numpy
from numpy import matlib

class RBF(IClassifier):
    def __init__(self, n_X, n_RBF, n_y):
        self._n_X = n_X
        self._n_RBF = n_RBF
        self._n_y = n_y
        self._Theta1 = matlib.randn((n_RBF, n_X))
        self._Theta2 = matlib.randn((n_RBF + 1, n_y))
    
    def fit(self, X, y):
        # here will train the model using X and y
    
    def predict(self, X):
        # here will give the prediction of X
        # X is supposed to be an matrix or 2-d array and each sample shall be shaped in a row vector
        
        for x in X:
    
    def test(self, X, y):
        # here will test with X and y