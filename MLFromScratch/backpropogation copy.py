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
        self.activation_function = np.vectorize(lambda x: max(0, x))
        self.output_activation_function = np.vectorize(sigmoid)
        self.loss_function = messi # implement MSE here
    

    def forward(self, x, train=False):
        # assume x is input size
        neurons = x
        activations = [neurons] # used in backpropagation
        z_values = []
        for i, layer in enumerate(layers[1:]):
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
        
        if train:
            return neurons, activations, z_values
        return neurons
    

    def __repr__(self):
        return f"Input size: {self.input_size}\n" +\
            f"Weights: {[weight_matrix.shape for weight_matrix in self.weights]}\n"    +\
            f"Biases: {[bias_vector.shape for bias_vector in self.biases]}\n"




    def train_on_single_batch(self, x, y_target):
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


        # do just weights for no
        # calculate dL/d_weights for each weight matix
        loss_gradients_W = []
        loss_gradients_B = []
        for i, previous_activation in enumerate(activations[::-1][1:]): # go through layers backwards, skipping the output
            previous_W = self.weights[::-1][i]
            previous_B = self.biases[::-1][i]
            if i == 0:
                # calculate dL/d_activation
                # assuming MSE
                dl_____d_output = 2*(y_hat - y_target)
                # dl/d_activation = residual from last loop
            # calculate d_activation/d_z
            # assuming RELU
            previous_z = z_values[::-1][i]
            d_activation_____d_z = np.vectorize(lambda z: 1 if z>0 else 0)(previous_z)

            # calculate d_z/d_W and d_z/d_B
            # literally just the activation bro
            d_z___d_w = previous_activation
            d_z___d_b = np.ones(previous_activation.shape) # of shape ???

            # multiply all of em together and add as the weight's loss gradient
            
            if i == 0:
                print("W input shape: ", (dl_____d_output*d_activation_____d_z).shape, d_z___d_w.shape) # dz___dw should be ((21,), (4,))
                print("B input shape: ", (dl_____d_output.shape), (d_activation_____d_z).shape, d_z___d_b.shape) # dz___dw should be ((21,), (4,))

                loss_gradient_wrt_W_at_current_layer = np.outer(dl_____d_output*d_activation_____d_z, d_z___d_w) # should take an nx1 (newer layer size) and an mx1 (last layer size/activation - dzdw) and produce an mxn matrix
                loss_gradient_wrt_B_at_current_layer = np.matmul(d_activation_____d_z, np.outer(dl_____d_output, d_z___d_b))

                print("W output shape:", loss_gradient_wrt_W_at_current_layer.shape)
                print("B output shape:", loss_gradient_wrt_B_at_current_layer.shape)
                print()
            else:
                print("W input shape: ", loss_gradient_wrt_W_at_current_layer.shape, d_activation_____d_z.shape, d_z___d_w.shape)
                print("B input shape: ", dl_____d_output.shape, d_activation_____d_z.shape)

                #loss_gradient_wrt_W_at_current_layer = np.matmul(loss_gradient_wrt_W_at_current_layer, np.outer(d_activation_____d_z, d_z___d_w))
                loss_gradient_wrt_W_at_current_layer = np.matmul(np.outer(d_activation_____d_z, d_z___d_w), loss_gradient_wrt_W_at_current_layer)
                loss_gradient_wrt_B_at_current_layer = np.matmul(loss_gradient_wrt_B_at_current_layer, np.outer(d_activation_____d_z, d_z___d_b))                                # dl_____d_output * d_activation_____d_z * d_z___d_b

                print("W output shape:", loss_gradient_wrt_W_at_current_layer.shape) # (21 4), (4, 13) (13 2) (2 7)
                print("B output shape:", loss_gradient_wrt_B_at_current_layer.shape) # (21)    (4)     (13)   (2)
                print()


            loss_gradients_W.append(loss_gradient_wrt_W_at_current_layer)
            loss_gradients_B.append(loss_gradient_wrt_B_at_current_layer)
            

        
        # go back over each weight/bias and add the gradient
        for i, weight_update in enumerate(loss_gradients_W[::-1]):
            # start from weights at the input layer
            weight_update = loss_gradients_W[::-1][i]
            print(self.weights[i].shape, weight_update.shape)
            self.weights[i] += weight_update 

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