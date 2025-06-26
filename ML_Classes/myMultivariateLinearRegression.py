import numpy as np
import matplotlib.pyplot as plt

class myMultivariateLinearRegression:
    # initialize variables
    def __init__(self, learning_rate=0.01, num_iterations=1000):       
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.intercept = None
        self.mean = None
        self.std = None
        self.standardized_X = None
        self.cost_history = None
    
    # standardize the data
    def standardize(self, X):        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.standardized_X = (X - self.mean) / self.std
        
        return self.standardized_X
    
    # cost function method
    def cost_function(self, X, y, theta):       
        m = len(y)
        predictions = np.dot(X, theta)
        error = predictions - y
        cost = (1 / (2 * m)) * np.sum(error**2)
        
        return cost

    # gradient descent method
    def gradient_descent(self, X, y):       
        m, n = X.shape
        self.theta = np.zeros(n)
        self.intercept = 0
        self.cost_history = []

        for _ in range(self.num_iterations):
            # Calculate predictions
            predictions = np.dot(X, self.theta) + self.intercept

            # Calculate errors
            errors = predictions - y

            # Update thetas and intercept using gradient descent
            self.theta -= (self.learning_rate / m) * np.dot(X.T, errors)
            self.intercept -= (self.learning_rate / m) * np.sum(errors)
            
            cost = self.cost_function(X, y, self.theta)
            self.cost_history.append(cost)
            
        return self.cost_history
    
    # fit method to train data to model
    # predict method to transorm data to model
    def fit_predict(self, X, y):
        # Standardize the features/ X values/ dependent variables 
        X = self.standardize(X)

        # Add a column of ones for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Perform gradient descent
        self.gradient_descent(X, y)

        # predict
        predictions = np.dot(X, self.theta) + self.intercept
        
        # return predicted y values 
        return predictions    
    
    # plot method for convergence v cost function
    def plot_convergence(self):
        plt.plot(range(1, self.num_iterations + 1), self.cost_history, color='blue')
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.grid()
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Cost Function')
        plt.show()
