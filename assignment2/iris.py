import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# f = open('IrisTrainML.dt', 'r')

# data = f.read()
# f.close

# print(data)

# Location varialbe for trainging and test data
LocationTrain = 'IrisTrainML.dt'
LocationTest = 'IrisTestML.dt'

# Reading data into dataframes
dfTrain = pd.read_csv(LocationTrain, header=None, sep=' ')
dfTest = pd.read_csv(LocationTest, header=None, sep=' ')

# Extracting X and Y from training and test data into matrices
IrisTrainX = dfTrain.as_matrix(columns=[0,1])
IrisTrainY = dfTrain.as_matrix(columns=[2])
IrisTestX = dfTest.as_matrix(columns=[0,1])
IrisTestY = dfTest.as_matrix(columns=[2])


#Computes the distance between two vectors/datapoints
def distance(vector1, vector2):
	return NP.linalg.norm(vector1 - vector2)

#Finds the vector in a matrix closest to a given vector
#Uses distance
def minDistance(matrix,vector):
	dmin = 100
	imin = 0
	for i in range(0,len(matrix)):
		d1 = distance(matrix[i],vector)
		if  d1 <= dmin:
			dmin = d1
			imin = i
		else:
			dmin = dmin
			imin = imin

	return imin



print(IrisTestY)


