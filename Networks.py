#Matrix library
import numpy

class FeedForwardNeuralNetwork:
	"""
	This is the constructor for a variable network class. Randomly initializes the 
	"""
	def __init__(self, size, activationFunction, activationDerivative, costFunction, costDerivative):
		self.layers = len(size) - 1
		#Initialize weights using a new standard deviation 1/sqrt(activations) to reduce early saturation of activation functions
		self.weights = [numpy.random.normal(0, numpy.power(x, -0.5), (y, x)) for x, y in zip(size[:-1], size[1:])]
		#Initialize biases to 0 and allow backprop to solve for values (set to 0.0001 for ReLUs activation function -> why?)
		self.biases = [numpy.zeros((x, 1)) for x in size[1:]]
		#Vectorize functions so they can be applied to matrices
		self.activationFunction = numpy.vectorize(activationFunction)
		self.activationDerivative = numpy.vectorize(activationDerivative)
		self.costFunction = numpy.vectorize(costFunction)
		self.costDerivative = numpy.vectorize(costDerivative)

	"""
	This is the feed forward algorithm for the neural network
	:param inputLayers: A matrix containing the input layer values for to be fed forward
	:param layer: An integer denoting the current layer being fed forward
	:return: A tuple containing the final output layer after being fed forward
	"""
	def feedforward(self, inputLayers):
		weightedSum = []
		activations = [inputLayers]
		for layer in range(self.layers):
			#Get weighted sum matrix (not yet activated)
			weightedSum.append(numpy.dot(self.weights[layer], numpy.array(activations[layer])) + self.biases[layer])
			#Activate the weighted sum using activationFunction
			activations.append(self.activationFunction(weightedSum[layer]))
		# return weightedSum, activations 
		return weightedSum, activations

		
	# """
	# This is the implementation for the stochastic gradient descent algorithm for learning
	# """
	# def stochasticGradientDescent(learningRate, regularizationParameter, trainingExamples, miniBatchSize, epoch):

	"""
	This is the backpropagation algorithm implemented via the 4 fundamental backpropagation algorithm
	Expected Outputs are the 
	"""
	def backpropagation(self, weightedSums, activations, trainingExamples):
		dweights = [];
		errors = [self.firstFundamentalEquation(weightedSums, activations, trainingExamples[1])]
		for layer in range(self.layers-1, 0, -1):
			errors.insert(0, self.secondFundamentalEquation(weightedSums, errors, layer))
		print(errors)
		# for layer in range(self.layers - 1):
		# 	dweights.append(self.thirdFundamentalEquation(activations, errors, layer))
		return dweights, errors

	"""
	This is the implementation for the first fundamental equation of backpropagation;
	a formula for the gradient of the last layer with respect to the weighted sum of the
	last layer
	"""
	def firstFundamentalEquation(self, weightedSums, activations, trainingExamples):
		#C'(activation L,y) * sigma'(activation L - 1)
		print(typeof())
		return self.costDerivative(activations[-1], trainingExamples[1]) * self.activationDerivative(weightedSums[-1])

	"""
	This is the implementation for the second fundamental equation of backpropagation;
	a formula for the gradient of any layer with respect to the weighted sum of the 
	next layer
	"""
	def secondFundamentalEquation(self, weightedSums, errors, layer):
		return (self.weights[layer].T @ errors[0]) * self.activationDerivative(weightedSums[layer-1]) 

	"""
	This is the implementation for the third fundamental equation of backpropagation;
	a formula for the gradient of any layer with respect to its weights
	"""
	def thirdFundamentalEquation(self, activations, errors, layer):
		dweights = []
		#check this, i think this should work, create a matrix with activations as rows multiplied by the error value going down


		dweights.append(numpy.dot(activations[layer] , errors[layer]))

		print(dweights)
		return dweights

	"""
	This is the implementation for the fourth fundamental equation of backpropagation;
	a formula for the gradient of any layer with respect to its biases. Here for the sake of completeness
	"""
	def fourthFundamentalEquation(self, error):
		return error





# """

# Variable cost functiosn and costDerivatives
# Variable System architecture
# MATRIX Feedforward/Backpropocation

# """


# """

# Python Data Science Stack
# Adversarial Learning

# Pandas efficient mechanism for data analysis

# Normalize datasat
# 1. ensure that a particular variable is within a specific space
# 2. ensure that all values are within a specific range

# """