import numpy
from numpy import matlib

from . import IClassifier
from utilities import *

class MLP:
    def __init__(self, *args):
        self._n_X = args[0]
        self._n_y = args[len(args) - 1]
        self._n_hidden = args[1:len(args) - 1]
        self._Theta = []
        
        for i in range(len(args) - 1):
            self.Theta.append(matlib.randn((args[i] + 1, args[i + 1])))

    def fit(self, X, y):
        # here will train the model using X and y
        pass
    
    def predict(self, X):
        # here will give the prediction of X
        pass
    
    def test(self, X, y):
        # here will test with X and y
        pass

IClassifier.register(MLP)