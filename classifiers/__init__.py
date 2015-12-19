import numpy
from numpy import matlib

from abc import ABCMeta, abstractmethod

from utilities import *

class IClassifier(metaclass=ABCMeta):
    """The abstract base class of all the Classifiers"""
    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        """Train the model using X and y
        
        X is the feature matrix, each sample shall be shaped in a row vector
        y is the target result, each row is whether belongs to this class
        Return the final J (value of the cost function)
        
        """
        pass
    
    @abstractmethod
    def predict(self, X, *args, **kwargs):
        """Give the prediction of X
        
        X is the feature matrix, each sample shall be shaped in a row vector
        Return a matrix, each row is the classification result of each sample
         
        """
        pass
    
    @abstractmethod
    def test(self, X, y, *args, **kwargs):
        """Test with X and y
        
        X is the feature matrix, each sample shall be shaped in a row vector
        y is the test result, each row is whether belongs to this class
        
        """
        pass

class RBF():
    def __init__(self, n_X, n_RBF, n_y):
        self._n_X = n_X
        self._n_RBF = n_RBF
        self._n_y = n_y
        self._Theta1 = matlib.randn((n_RBF, n_X))
        self._Theta2 = matlib.randn((n_RBF + 1, n_y))
    
    def fit(self, X, y, *args, **kwargs):
        pass
    
    def predict(self, X, *args, **kwargs):
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
        return sigmoid(raw_output) > 0.5
    
    def test(self, X, y, *args, **kwargs):
        # now return the number of wrong cases
        y_predict = self.predict(X)
        return numpy.count_nonzero(numpy.sum(y_predict != y, axis=1))

class MLP:
    def __init__(self, *args):
        if len(args) < 2:
            raise TypeError("__init__() missing 2 required positional arguments: 'n_X' and 'n_y'")
        
        self._n = args
        self._Theta = []
        for i in range(len(args) - 1):
            self._Theta.append(matlib.randn((args[i] + 1, args[i + 1])))
        
    def fit(self, X, y, *args, **kwargs):
        pass
    
    def predict(self, X, *args, **kwargs):
        m = X.shape[0]
        Z = None
        S = None
        A = numpy.concatenate((matlib.ones((m, 1)), X), axis=1)
        
        is_first_layer = True
        for Theta_index in range(len(self._Theta)):
            if is_first_layer:
                is_first_layer = False
            else:
                A = numpy.concatenate((matlib.ones((m, 1)), Z), axis=1)
            
            Z = A * self._Theta[Theta_index]
            S = sigmoid(Z)
        
        return S > 0.5
    
    def test(self, X, y, *args, **kwargs):
        # now return the number of wrong cases
        y_predict = self.predict(X)
        return numpy.count_nonzero(numpy.sum(y_predict != y, axis=1))

IClassifier.register(MLP)
IClassifier.register(RBF)