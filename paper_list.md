# ML Paper List
goal is to get my bearings in ML

paper "completed" date is whenever I have met the following criteria
- have a mental model solid enough to allow me to implement the paper from scratch in under 30mins
- write a useful explanation in under 200 words
	
paper mastery is whenever I have met the following criteria
- have a mental model solid enough to allow me to implement the paper from scratch in under 10mins
- write a useful explanation in under 15 words


---

### 📚 Foundational Papers
- [ ] "A Logical Calculus of the Ideas Immanent in Nervous Activity" by McCulloch and Pitts (1943)
- [ ] "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" by Rosenblatt (1958)
- [ ] "Learning Representations by Back-propagating Errors" by Rumelhart, Hinton, and Williams (1986)
- [ ] "Induction of Decision Trees" by Quinlan (1986)
- [ ] "Handwritten Digit Recognition with a Back-Propagation Network" by LeCun et al. (1989)

### 🧠 Neural Networks and Deep Learning
- [ ] "Deep Learning" by LeCun, Bengio, and Hinton (2015)
- [ ] "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky, Sutskever, and Hinton (2012)
- [ ] "Sequence to Sequence Learning with Neural Networks" by Sutskever, Vinyals, and Le (2014)
- [ ] "Generative Adversarial Networks" by Goodfellow et al. (2014)
- [ ] "Deep Residual Learning for Image Recognition" by He et al. (2015)

### 🕹️ Reinforcement Learning
- [ ] "Q-Learning" by Watkins (1989)
- [ ] "Policy Gradient Methods for Reinforcement Learning with Function Approximation" by Sutton et al. (2000)
- [ ] "Playing Atari with Deep Reinforcement Learning" by Mnih et al. (2013)
- [ ] "Human-level Control through Deep Reinforcement Learning" by Mnih et al. (2015)
- [ ] "Mastering the Game of Go with Deep Neural Networks and Tree Search" by Silver et al. (2016)

### 🗣️ Natural Language Processing
- [ ] "A Statistical Approach to Machine Translation" by Brown et al. (1990)
- [ ] "Attention Is All You Need" by Vaswani et al. (2017)
- [ ] "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2019)
- [ ] "Language Models are Few-Shot Learners" by Brown et al. (2020)
- [ ] "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Raffel et al. (2020)

### 🧪 Unsupervised and Semi-Supervised Learning
- [ ] "Reducing the Dimensionality of Data with Neural Networks" by Hinton and Salakhutdinov (2006)
- [ ] "Unsupervised Feature Learning and Deep Learning: A Review and New Perspectives" by Bengio et al. (2013)
- [ ] "Semi-Supervised Learning with Ladder Networks" by Rasmus et al. (2015)
- [ ] "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford, Metz, and Chintala (2016)
- [ ] "Improved Techniques for Training GANs" by Salimans et al. (2016)

### 🔍 Bayesian Methods and Probabilistic Models
- [ ] "Bayesian Learning for Neural Networks" by MacKay (1992)
- [ ] "Practical Bayesian Framework for Backpropagation Networks" by Neal (1996)
- [ ] "Latent Dirichlet Allocation" by Blei, Ng, and Jordan (2003)
- [ ] "Bayesian Reasoning and Machine Learning" by Barber (2012)
- [ ] "Auto-Encoding Variational Bayes" by Kingma and Welling (2014)

---

# Paper notes

## Attention is all you need (start-end: 4/?? - ???)
Goal: be able to write a transformer from scratch as fast as possible
inference and train

speedrun up to this point then work on understnaing


### Basic overview

### Fundamentals to compress
Vanishing gradient problem and residuals
	- f(input) = combination(input, change) (i.e. the transformed state of the input) is more complex to calculate than g(input) = change
	- therefore networks more efficiently train to calculate g
	- therefore if you need f(input), just set you model up to calculate g, then manually do f(input) = combination(input, g(input)
backpropagation and gradient of the loss
	- loss gradient is the multidimensional direction that the loss goes in
cross entropy loss
	- Dot({output probabilities}{correct 'probabilities' - in this case, a vector of all embeddings with 1 for the target token and 0 for all other tokens})
	- Essentially asks the question: how accurate is your model's predicted probability distribution to the real target distribution from the training data?



## General ML notes
Divergence
	- the difference between probability distributions (example KL divergence)
Adam
	- A type of optimizer
Matrix multiplication
	- a @ b
	- each vector of b is transformed one at a time, feeding into paper-shredder a
Chain rule for backprop
	- Tracks a "chain" of changes between variables