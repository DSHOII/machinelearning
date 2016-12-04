import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


#___________________________Zero-one loss kNN-model____________________________#

# Location variable for training and test data
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


# Computing error on training and test data for 1NN, 3NN and 5NN
# train1NN = zeroOneLoss(irisTrainX, irisTrainY, irisTrainX, irisTrainY, 1)
# train3NN = zeroOneLoss(irisTrainX, irisTrainY, irisTrainX, irisTrainY, 3)
# train5NN = zeroOneLoss(irisTrainX, irisTrainY, irisTrainX, irisTrainY, 5)
# test1NN = zeroOneLoss(irisTrainX, irisTrainY, irisTestX, irisTestY, 1)
# test3NN = zeroOneLoss(irisTrainX, irisTrainY, irisTestX, irisTestY, 3)
# test5NN = zeroOneLoss(irisTrainX, irisTrainY, irisTestX, irisTestY, 5)


# Printing error results
# print('Number of training cases is ' + str(len(irisTrainX)))
# print('Number of test cases is ' + str(len(irisTestX)) + '\n')

# print('1-nearest neighbor training error: ' + str(train1NN))
# print('3-nearest neighbor training error: ' + str(train3NN))
# print('5-nearest neighbor training error: ' + str(train5NN))
# print('1-nearest neighbor testing error: ' + str(test1NN))
# print('3-nearest neighbor testing error: ' + str(test3NN))
# print('5-nearest neighbor testing error: ' + str(test5NN))


#______________________5-fold cross-validation________________________________#
# This solution is largely hard coded, since I could not come up with a good
# way of iterate through both he 5-fold cross validation and the different k's.


# General data splitting function for crossvalidation. Dataframe as input
# Assumes category/class as the single, last coloumn
def splitter(dataframe, k):
        vectorLength = ((dataframe.shape)[1]) - 1
        dataMatrix = dataframe.as_matrix()
        splitMatrix = np.array_split(dataMatrix, k)
        return splitMatrix


# Combining split data back into new traning and test sets
splitMatrix = splitter(dfTrain, 5)

cVTrain1 = np.vstack((splitMatrix[0], splitMatrix[1],
                            splitMatrix[2], splitMatrix[3]))

cVTest1 = splitMatrix[4]

cVTrain2 = np.vstack((splitMatrix[1], splitMatrix[2],
                            splitMatrix[3], splitMatrix[4]))

cVTest2 = splitMatrix[0]

cVTrain3 = np.vstack((splitMatrix[2], splitMatrix[3],
                            splitMatrix[4], splitMatrix[0]))

cVTest3 = splitMatrix[1]

cVTrain4 = np.vstack((splitMatrix[3], splitMatrix[4],
                            splitMatrix[0], splitMatrix[1]))

cVTest4 = splitMatrix[2]

cVTrain5 = np.vstack((splitMatrix[4], splitMatrix[0],
                            splitMatrix[1], splitMatrix[2]))

cVTest5 = splitMatrix[3]


# Creating a variable to hold results of 5-fold cross validation
resM = np.zeros([26, 5])


# A single fold of the k-nearest neighbor cross validation.
def kNNCross():
        for i in range(1, 26, 2):
                resM[i,0] = zeroOneLoss(cVTrain1[:,[0,1]], cVTrain1[:,[2]],
                                        cVTest1[:,[0,1]], cVTest1[:,[2]], i)
                resM[i,1] = zeroOneLoss(cVTrain2[:,[0,1]], cVTrain2[:,[2]],
                                        cVTest2[:,[0,1]], cVTest2[:,[2]], i)
                resM[i,2] = zeroOneLoss(cVTrain3[:,[0,1]], cVTrain3[:,[2]],
                                        cVTest3[:,[0,1]], cVTest3[:,[2]], i)
                resM[i,3] = zeroOneLoss(cVTrain4[:,[0,1]], cVTrain4[:,[2]],
                                        cVTest4[:,[0,1]], cVTest4[:,[2]], i)
                resM[i,4] = zeroOneLoss(cVTrain5[:,[0,1]], cVTrain5[:,[2]],
                                        cVTest5[:,[0,1]], cVTest5[:,[2]], i)



        return resM[[1,3,5,7,9,11,13,15,17,19,21,23,25],:]

# Computing average loss for each fold




brit = kNNCross()
print(brit)

john = cVTest5[:,[2]]
carsten = cVTest5[:,[0,1]]

