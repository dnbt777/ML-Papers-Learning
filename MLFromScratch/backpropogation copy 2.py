import numpy as np


class MLP():
    def __init__(self, layers):
        self.input_size = layers[0]
        self.weights = [
            np.random.randn(*(dim_out, dim_in)) for dim_in, dim_out in zip(layers[:-1], layers[1:])
        ]
        self.biases = [
            np.random.randn(*(n,)) for n in layers[1:]
        ]
        self.layers = [np.zeros((layer_size,)) for layer_size in layers]
        self.activation_function = np.vectorize(lambda x: max(0, x))
        self.output_activation_function = np.vectorize(sigmoid)
        self.loss_function = messi # implement MSE here
    

    def forward(self, x, train=False):
        # assume x is input size
        neurons = x
        activations = [neurons] # used in backpropagation
        self.layers[0] = np.array(neurons)
        z_values = []
        for i, layer in enumerate(self.layers[1:]):
            weights = self.weights[i]
            biases = self.biases[i]
            z = np.matmul(weights, neurons)
            z = z + biases
            z_values.append(z) # used for backpropogation
            if i == len(layers) - 1:
                neurons = self.output_activation_function(z)
            else:
                neurons = self.activation_function(z)
            activations.append(neurons) # used in backpropogation to calculate d_Z/d_W
        # calculate output activations
            self.layers[i] = np.array(neurons)
        
        if train:
            return neurons, activations, z_values
        return neurons
    

    def __repr__(self):
        return f"Input size: {self.input_size}\n" +\
            f"Weights: {[weight_matrix.shape for weight_matrix in self.weights]}\n"    +\
            f"Biases: {[bias_vector.shape for bias_vector in self.biases]}\n"




    def train_on_single_batch(self, x, y_target, eta=1e-4):
        y_hat, activations, z_values = self.forward(x, train=True)
        loss = self.loss_function(y_hat, y_target)
        # backpropogation starts here
        # start with last weights and biases
        # goal: find d_loss / d_weights, and d_loss / d_biases
        # gonna start by listing the partial derivatives of each component
        # loss = target - output
            # output = relu(weights_n(activation_n-1))    # relu or sigmoid, whatever
        # d_loss / d_weights = d_loss / activation_n * output_n / d_weights
        #
        # d_L/d_W = d_L/d_output * d_output/d_z * d_z/d_W
        # d_loss / d_output = 2*(output_y - target_y) # if MSE
        # d_output/d_z = ??? (depends on the activation function) if relu: {1 if z>0, 0 otherwise}
        #     d_output/d_z sigmoid: sigmoid(z)*(1-sigmoid(z))
        #     d_output/d_z tanh: 1 - tanh^2(z)
        #     d_output/d_z linear (no activation): 1
        # d_z/d_W = layer_n-1

        # Something like this I think
        # d_L___d_output = 2*(output_y - target_y)
        # d_z___d_W = x # its just the last layer bro
        # activation = "sigmoid"
        # if activation=="relu":
        #     pass
        # if activation=="sigmoid":
        #     d_output_over_d_z


        # Calculate initial chain links for output layer
        dL_dW = [np.zeros(weight.shape) for weight in self.weights]
        dL_dB = [np.zeros(bias.shape) for bias in self.biases]

        dL_da = 2*(y_hat - y_target) # MSE (layer_n,)
        da_dz = sigmoid(z_values[-1])*(1-sigmoid(z_values[-1])) # sigmoid (layer_n,)
        dL_dz = dL_da * da_dz # (layer_n,)

        dz_dw = activations[-2] # (layer_n-1,) (is this dz dw? have an llm explain)
        dL_dW[-1] = np.dot(dL_dz[:, np.newaxis], dz_dw[np.newaxis, :]) # (1,layer_n-1) dot (layer_n,) => (layer_n-1, layer_n) which is the size of the weights from n-1 to n
        dL_dB[-1] = dL_dz

        # continue for each hidden layer
        for i in range(len(self.layers) - 2, -1, -1):
            da_dz = relu_derivative(z_values[i])
            if i > 0:
                dz_dW = activations[i - 1]
            else:
                dz_dW = x # First layer

            dL_dw = np.dot(dL_dz[:, np.newaxis], dz_dW[np.newaxis, :])

            dL_dW[i] = dL_dw
            dL_dB[i] = dL_dz

            if i > 0:
                w = self.weights[i]
                dL_dz = np.dot(w.T, dL_dz) * relu_derivative(z_values[i])

        
        # go back over each weight/bias and add the gradient
        for i, gradients in enumerate(zip(dL_dW, dL_dB)):
            dW, dB = gradients
            # start from weights at the input layer
            weight_update = dW
            bias_update = dB
            print(weight_update.shape, self.weights[i].shape)
            self.weights[i] -= eta*weight_update 
            self.biases[i] -= eta*bias_update 

            # start w biases from the input layer
            #bias_update = loss_gradients_B[::-1][i]
            #self.biases[i] += bias_update



        # if a bunch of changes lead from y to x, you can calculate how much dx effects dy by chaining together all those changes (WTF!)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def messi(y_hat, y_target):
    assert len(y_hat) == len(y_target), "MSE: Y lengths do not match"
    n = len(y_hat)
    mse = np.vectorize(lambda yhat, ytarget: (ytarget-yhat)**2)(y_hat, y_target)
    mse = sum(mse)/n
    return mse

@np.vectorize
def relu_derivative(x):
    if x > 0:
        return 1
    return 0


if False:
    mse_test = messi([1, 2, 4], [1, 1, 1])
    print(mse_test)

if False:
    layers = [7, 2, 13, 4, 21]
    model = MLP(layers)
    x = np.ones((layers[0],))
    print(model)
    y = model.forward(x)
    print("Example forward pass output:", y)


print("Begin training run")
# goal should be to have the model output all ones (for now!)
layers = [7, 2, 13, 4, 21]
model = MLP(layers)
x = np.ones((layers[0],))
y_target = np.ones(layers[-1],)
print(model)
# start training
model.train_on_single_batch(x, y_target)



# tactic
# write out my current understanding of {paper}, then ask the model to correct it.