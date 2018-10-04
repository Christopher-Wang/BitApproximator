# Example of kNN implemented from Scratch in Python
import csv
import random
import math
import operator

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

class kNN(object):
	def __init__(self, k, distance, data):
		self.k = k
		self.distance = distance
		self.data = data

	def getNeighbors(self, testInstance, columns):
		distances = []
		for coloumn in columns:
			dist =self.distance(testInstance, self.data[x], length)
			distances.append((self.data[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(self.k):
			neighbors.append(distances[x][0])
		return neighbors
	 
	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]
	 
