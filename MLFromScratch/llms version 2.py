import numpy as np

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = {}
        self.cache = {}
        self.grads = {}
        
        # Initialize weights and biases
        for i in range(1, len(layers)):
            self.parameters['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((layers[i], 1))
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        self.cache['A0'] = X
        A = X
        
        for i in range(1, len(self.layers)):
            W = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]
            Z = np.dot(W, A) + b
            A = self.sigmoid(Z)
            
            self.cache['Z' + str(i)] = Z
            self.cache['A' + str(i)] = A
            
        return A
    
    def compute_cost(self, Y_hat, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return cost
    
    def backward_propagation(self, Y):
        m = Y.shape[1]
        L = len(self.layers) - 1
        
        # Initialize backpropagation
        Y_hat = self.cache['A' + str(L)]
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        
        for l in reversed(range(1, L+1)):
            dA_curr = dA_prev
            A_prev = self.cache['A' + str(l-1)]
            Z_curr = self.cache['Z' + str(l)]
            W_curr = self.parameters['W' + str(l)]
            b_curr = self.parameters['b' + str(l)]
            
            m = A_prev.shape[1]
            dZ_curr = dA_curr * self.sigmoid_derivative(Z_curr)
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            dA_prev = np.dot(W_curr.T, dZ_curr)
            
            self.grads['dW' + str(l)] = dW_curr
            self.grads['db' + str(l)] = db_curr
    
    def update_parameters(self, learning_rate=0.01):
        L = len(self.layers) - 1
        for l in range(1, L+1):
            self.parameters['W' + str(l)] -= learning_rate * self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * self.grads['db' + str(l)]

# Example usage:
np.random.seed(1)
mlp = MLP([10, 4, 5, 6, 6, 10])
X = np.random.randn(10, 5)
Y = np.random.randint(0, 2, (10, 5))

# Forward propagation
Y_hat = mlp.forward_propagation(X)

# Compute cost (for binary classification)
cost = mlp.compute_cost(Y_hat, Y)

# Backward propagation
mlp.backward_propagation(Y)

# Update parameters
mlp.update_parameters(learning_rate=0.01)