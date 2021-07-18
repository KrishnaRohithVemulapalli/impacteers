from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = load_iris()
X = data.data                               #Features of the flowers
y = data.target                             #Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class LinearRegression:
    #A class which implements linear regression model with gradient descent.
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        ##Initialising our required variables
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None
        self.loss = []
    def _mean_squared_error(self,y, y_hat):
        '''
         method used to evaluate loss at each iteration.

        :param: y - array, true values
        :param: y_hat - array, predicted values
        :return: float
        '''
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) ** 2
        return error / len(y)
    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the linear regression model.

        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            # Line equation
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict(self, X):
        '''
        Makes predictions using the line equation.

        :param X: array, features
        :return: array, predictions
        '''
        return np.dot(X, self.weights) + self.bias



def solve_linearregression(x1,x2,x3,x4):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = r2_score(y_test,preds)

    answer=np.round(model.predict([[x1,x2,x3,x4]]))
    if answer == 0:
        string = 'Iris-setosa'
    elif answer == 1:
        string = 'Iris-versicolor'
    else:
        string = 'Iris-virginica'
    return (acc,string)