# Example of kNN implemented to classify food groups from Open Government nutrition data
# open.canada.ca/data/en/dataset/a289fd54-060c-4a96-9fcf-b1c6e706426f
import csv
import pandas as pd
import numpy as np
import os

#Gets the title headers from .csv files (They are denoted by being in all caps in the first row)
def getHeaders(filename):
	headers = [0, 1, 3]
	with open(filename, 'rt') as csvfile:
		read = csv.reader(csvfile)
		read = list(read)
		for i, row in enumerate(read):
		 if row[0].isupper():
		 	headers.append(i)
	return headers

#Get a dataframe from a datafile(+path)
def getDataFrame(dataFile):
	#Get the headers to skip
	headers = getHeaders("Data_Food/" + dataFile)
	df = pd.read_csv("Data_Food/" + dataFile, skiprows = headers)
	#Add a class column
	df["Class"] = dataFile[:-4]
	return df

#Get the concatenated data frames from all .csvfiles
def getDataFrames(dataFiles):
	dfs = []
	for dataFile in dataFiles:
		dfs.append(getDataFrame(dataFile))
	return pd.concat(dfs, join="inner")

#Clean the data
def cleanDataFrame(dataframe, tr = 0.001, nan = 0):
	dataframe = dataframe.set_index('Food name')
	dataframe = dataframe.replace("tr", tr)
	dataframe[dataframe.columns[1:-1]] = dataframe[dataframe.columns[1:-1]].apply(pd.to_numeric)
	return dataframe.replace(np.nan, nan)

#Return the training and testing data, currently taken randomly
def getData():
	dataFiles = [file for file in os.listdir("Data_Food/")]
	dataframe = getDataFrames(dataFiles)
	dataframe = cleanDataFrame(dataframe)
	return dataframe

dataframe = getData()
mask = np.random.rand(len(dataframe)) < 0.8
trainingData, testingData = dataframe[mask], dataframe[~mask]
npval = dataframe.get_values()
print(npval[100])
