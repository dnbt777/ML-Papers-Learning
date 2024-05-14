# Experiments

ML experiments I have run and their results


# Transformer training data compression

## Hypothesis
Pre-compression of data improves transformer learning


## Method
Make string training data out of a small vocab. Duplicate the training set and gzip each string. Train a different transformer on each set. Measure learning rates


## Results
-- show chart here ---

The transformer learned faster, but there are caveats


## Conclusion
finish this




# LHTL MLP network


## Idea
A typical MLP has parameters for its network (W_MLP, B_MLP)
rather than have a learning rate, have a NN trained network for each parameter = W_LHTL)
update to the parameters is matmul(W_LHTL, loss_gradient)
in addition to training the network, train W_LHTL
do this by having a genetic algorithm that does 10 epochs of the MLP on 100 copies with random initialized weights to start off with (gen 0)
each next generation's weights will be a mutation of the previous generations' weights, weighted by fitness of course

wait, why tf am I using a matrix for W_LHTL? LHTL should be an MLP
	- the increase in loss is


differential learn-how-to-learn MLPs
MLP0 is the base MLP. (MLP(x) => y) => loss => loss gradient. this is what is used for inference.
	the loss gradient is passed through MLPn, then MLPn-1, MLP2, then MLP1, then the output of MLP1 is the matrix used to update the parameters of the base MLP, MLP0. (traditionally, though not in this architecture, parameters are trained via the loss and updated via the loss gradient)
MLP1 is an MLP. it takes in a matrix and outputs parameter updates, which are used to directly update MLP0.
MLP2 is an MLP. it takes in a matrix (flattened) and outputs
MLP...n is the final MLP. it takes the loss gradient as an input. it outputs something. it is trained using the nth derivative of the learning rate as a loss function. input layer size is the length of the flattened loss gradient matrix.


PARAMETERS:
MLP derivative count (n, above)
MLP0 inputs, hidden layers, and output layers
MLP{1-n} inputs, hidden layers, and output layers
	note: MLPn's input size, and MLP1's output size, are not determined by this parameter, but rather are both the size of the (flattened) loss gradient matrix. flattened in this context means that the matrix is turned into a vector of shape (rows*columns,)





Other ideas
transformer which takes MLP parameters as tokens and builds MLPs.. could this be useful? dont think so tbh