import numpy as np

class myNeuralNetwork:
    # initialize values
    def __init__(self,x,y,learning_rate,iters):
        np.random.seed(100)  
        
        self.x = x
        
        # standardize the data       
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.input = (self.x - self.mean) / self.std

        self.theta_1 = np.random.rand(self.input.shape[1],2) 
        self.theta_2 = np.random.rand(2, 1)              
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.learning_rate = learning_rate
        self.iters = iters
        self.cost = []
        self.bias = 1
        
    
    # sigmoid method   
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # find sigmoid derivative
    def sigmoid_derivative(self, x): 
        return x * (1 - x)

    # find the cost
    def cost_function(self, target, output):
        return 0.5 * np.sum(np.square(np.subtract(target, output)))
    
    # forward propogation method
    def forward_prop(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.theta_1) + self.bias)
        self.output = self.sigmoid(np.dot(self.layer1, self.theta_2) + self.bias)
    
    # backward propogation method
    def backward_prop(self):
        # application of the chain rule to find derivative of the cost function with respect to weights2 and weights1
        deriv_theta_2 = np.dot(self.layer1.T, ((self.y - self.output) * self.sigmoid_derivative(self.output)))
        deriv_theta_1 = np.dot(self.input.T, (np.dot((self.y - self.output) * self.sigmoid_derivative(self.output), self.theta_2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the cost function
        self.theta_1 += self.learning_rate * deriv_theta_1
        self.theta_2 += self.learning_rate * deriv_theta_2    
    
    # training method for the data utilizing forward and back propogation
    def train(self):
        for i in range (self.iters):
            self.forward_prop()
            self.backward_prop()
            
            self.cost.append(self.cost_function(self.y, self.output))
    
    # predict method to predict classification
    def predict(self,input_data):
        mean = np.mean(input_data, axis=0)
        std = np.std(input_data, axis=0)
        self.input = (input_data - mean) / std
        
        self.forward_prop()
        return np.round(self.output)
