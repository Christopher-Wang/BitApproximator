# Example of kNN implemented from Scratch in Python
import csv
import pandas as pd
import numpy as np
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

data = pd.concat(dfs, join="inner")
data = data.set_index('Food name')
data = data.replace("tr", 0.001)
data = data.replace(np.nan, 0)
print(data)
print(data.loc[data['Protein'] == "tr"])
# result = set(things[0])
# for s in things[1:]:
#     result.intersection_update(s)
# print(result)
