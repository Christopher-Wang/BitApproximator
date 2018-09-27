# Example of kNN implemented from Scratch in Python
import csv
import pandas as pd
import os

things = []
headerList = []
def loadDataset(filename):
	headers = [0, 1]
	with open(filename, 'rt') as csvfile:
		read = csv.reader(csvfile)
		read = list(read)
		for i, row in enumerate(read):
		 if row[0].isupper():
		 	headers.append(i)
	headerList.append(headers)


datasets = [file for file in os.listdir("Data_Food/")]
dfs = []
for i, data in enumerate(datasets):
	loadDataset("Data_Food/" + data)
	df = pd.read_csv("Data_Food/" + data, skiprows = headerList[i])
	dfs.append(df)

print(dfs[2])
# result = set(things[0])
# for s in things[1:]:
#     result.intersection_update(s)
# print(result)
