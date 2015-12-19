from abc import ABCMeta, abstractmethod

class IClassifier(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        # here will train the model using X and y
        pass
    
    @abstractmethod
    def predict(self, X):
        # here will give the prediction of X
        pass
    @abstractmethod
    def test(self, X, y):
        # here will test with X and y
        pass