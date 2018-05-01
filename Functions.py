import numpy
"""
This is an implementation of a sigmoid activation function. 
param: weightedSum a weightedSum to be activated
"""
#@staticmethod
def sigmoid(weightedSum):
	return 1/(1 + numpy.exp(-weightedSum))

"""
This is an implementation of the derivative of a sigmoid activation function. 
param: weightedSum a weightedSum to be derived
"""
# @staticmethod
def sigmoidPrime(weightedSum):
	activations = sigmoid(weightedSum)
	return activations * (1 - activations)

"""
This is an implementation of a quadratic cost function. 
param: activation The activated value of the output
param: expected The expected value of the output
"""
# @staticmethod
def quadratic(activation, expected):
	return 0.5 * numpy.square(activation - expected)

"""
This is an implementation of the derivative of a quadratic cost function. 
param: activation The activated value of the output
param: expected The expected value of the output
"""
# @staticmethod
def quadraticPrime(activation, expected):
	return 0.5 * numpy.square(activation - expected)