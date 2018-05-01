import Networks, Functions, numpy
testData = [
	[[0,0,0],[0,0]], 
	[[0,0,1],[0,1]], 
	[[0,1,0],[0,1]], 
	[[0,1,1],[1,0]], 
	[[1,0,0],[0,1]], 
	[[1,0,1],[1,0]], 
	[[1,1,0],[1,0]], 
	[[1,1,1],[1,1]] 
]



size = [3,4,4,2]
len(size)
thing = Networks.FeedForwardNeuralNetwork(size, Functions.sigmoid, Functions.sigmoidPrime, Functions.quadratic, Functions.quadraticPrime)
lol = numpy.array([testData[0][0], testData[1][0], testData[2][0]]).T
lmao = numpy.array([testData[0][1], testData[1][1], testData[2][1]]).T
x = thing.feedforward(lol)
y = thing.backpropagation(x[0], x[1], lmao)

#print(x[1])
