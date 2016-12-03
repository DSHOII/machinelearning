import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# f = open('IrisTrainML.dt', 'r')

# data = f.read()
# f.close

# print(data)

# Location varialbe for trainging and test data
locationTrain = 'IrisTrainML.dt'
locationTest = 'IrisTestML.dt'

# Reading data into dataframes
dfTrain = pd.read_csv(locationTrain, header=None, sep=' ')
dfTest = pd.read_csv(locationTest, header=None, sep=' ')

# Extracting X and Y from training and test data into matrices
irisTrainX = dfTrain.as_matrix(columns=[0,1])
irisTrainY = dfTrain.as_matrix(columns=[2])
irisTestX = dfTest.as_matrix(columns=[0,1])
irisTestY = dfTest.as_matrix(columns=[2])


# Computes the distance between two vectors/datapoints
def distance(vector1, vector2):
	return np.linalg.norm(vector1 - vector2)

# Finds the vector in a matrix closest to a given vector
# Uses distance
def minDistance(matrix, vector):
	dMin = 100
	iMin = 0
	for i in range(0,len(matrix)):
		d1 = distance(matrix[i], vector)
		if  d1 <= dMin:
			dMin = d1
			iMin = i
		else:
			dMin = dMin
			iMin = iMin
	return iMin


# Creates a sorted list of distances from a vector to all vectors of a matrix.
def sortDistMatrix(trainMatrixX, trainMatrixY, vectorX):
        dimensions = trainMatrixX.shape
        length = dimensions[0]
        distanceMatrix = np.zeros([length, 2])
        for i in range(0, length):
                dist = distance(trainMatrixX[i], vectorX)
                vector = np.array([dist, trainMatrixY.item(i)])
                distanceMatrix[i] = vector
        sortedDistanceMatrix = np.sort(distanceMatrix, axis=0)
        sortedDistanceMatrix = distanceMatrix[np.lexsort(np.fliplr(distanceMatrix).T)]
        return sortedDistanceMatrix


# General k-nearest neighbor classifier
def kNN(trainMatrixX, trainMatrixY, vectorX, k):
        sortedDistances = sortDistMatrix(trainMatrixX, trainMatrixY, vectorX)
        categoryList = np.zeros([k], dtype=int)
        for i in range(0, k):
                nnCategory = int(sortedDistances.item(i, 1))
                categoryList[i] = nnCategory
        counts = np.bincount(categoryList)
        classifiedAs = np.argmax(counts)
        return classifiedAs

# Function that finds the number of wrong classifications for test data
def testingFalse(matrix1X, matrix1Y, matrix2X, matrix2Y, k):
	trueCount = 0
	falseCount = 0
	for i in range(0,len(matrix2X)):
		if kNN(matrix1X, matrix1Y, matrix2X[i], k) == matrix2Y.item(i):
			trueCount += 1
		else:
			falseCount += 1

	return falseCount


# Finding the empirical loss of k-NN model  with a zero-one loss function
def zeroOneLoss(matrix1X, matrix1Y, matrix2X, matrix2Y, k):
        N = len(matrix2X)
        misClassified = testingFalse(matrix1X, matrix1Y, matrix2X, matrix2Y, k)
        zeroOneLoss = (1/N)*misClassified
        return zeroOneLoss



john = zeroOneLoss(irisTrainX, irisTrainY, irisTestX, irisTestY, 5)
print(john)

# print(len(irisTrainX))
# print(len(irisTrainY))
# print(len(irisTestX))
# print(len(irisTestY))
