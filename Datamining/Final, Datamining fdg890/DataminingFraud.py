#Importing numpy and sqlite3
import numpy as NP
import sqlite3 as sql
from numpy.linalg import *
import matplotlib.pyplot as mpl

#Establishing connection to the database and creating cursor
conn = sql.connect('DataMiningAssignment2015.db')
c = conn.cursor()

#Importing Keystrokes_Train to matrices
KeyTrainX = NP.matrix(c.execute("SELECT * FROM Keystrokes_Train_X;").fetchall())
KeyTrainY = NP.matrix(c.execute("SELECT * FROM Keystrokes_Train_Y;").fetchall())
KeyTestX = NP.matrix(c.execute("SELECT * FROM Keystrokes_Test_X;").fetchall())
KeyTestY = NP.matrix(c.execute("SELECT * FROM Keystrokes_Test_Y;").fetchall())

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

#1NN model: Compares dependable variables between two matrices.
#Uses minDistance
def compareY(matrix1X, matrix2X,matrix1Y,matrix2Y):
	trueCount = 0
	falseCount = 0
	for i in range(0,len(matrix2X)):
		if matrix1Y[minDistance(matrix1X,matrix2X[i])] == matrix2Y[i]:
			trueCount += 1
		else:
			falseCount += 1

	return trueCount

#Computes accuracy for 1NN model
def accuracy1NN(trueCount, datasetN):
	return trueCount / len(datasetN) * 100


OneNN1 = minDistance(KeyTrainX, KeyTrainX[336])
OneNN2 = compareY(KeyTrainX,KeyTestX,KeyTrainY,KeyTestY)
OneNN3 = accuracy1NN(OneNN2,KeyTestX)

print ("Perfomance of 1NN:")
print (OneNN1)
print (OneNN2)
print (OneNN3)
print ("\n")

#Computes covariance matrix of KeyTrainX transposed
KeyTrainXCov = NP.cov(KeyTrainX.T)

#Computes eigen values and eigen vectors of the covariance matrix
eVal, eVec = NP.linalg.eigh(KeyTrainXCov)

#reorders the eigen values in decreasing order
eValOrder = eVal[::-1]

#Computing total sum of eigen values
eValTotalSum = NP.sum(eVal)

#Sums a specified number of values in a matrix from top down
def sumInc(eigenvalues, count):
	sumCount = 0
	eSum = 0.0
	for i in range(0,count):
		if sumCount < count:
			eSum += NP.sum(eigenvalues[sumCount])
			sumCount += 1

	return eSum

#PCA model computing the explained variance from a number of components
#Uses sumInc
def PCA(eigenvalues, desiredExplainedPrc):
	compCount = 0
	explainedVar = 0.0
	for i in range(0,len(eigenvalues)):
		if sumInc(eigenvalues,compCount) / eValTotalSum * 100 < desiredExplainedPrc:
			explainedVar = sumInc(eigenvalues,compCount+1) / eValTotalSum * 100
			compCount += 1
		else:
			compCount

	return compCount, explainedVar

print ("Performance of PCA:")
print (PCA(eValOrder,90))
print ("\n")

#Creates an array to go as X-axis in the scatter plot
xAxis = NP.arange(1,22)


#Plotting eigenspectrum
mpl.figure(1)
mpl.plot(xAxis,eValOrder,"bo")
mpl.xlim([0,23])
mpl.ylim([0,0.06])
mpl.xlabel("Eigenvalue number")
mpl.ylabel("Eigenvalue")
mpl.title("Eigenspectrum")
mpl.show()


#Creating array of first two principal components
principalTwo = NP.vstack((eVec[:,20],eVec[:,19]))

#Computing data base shifted by principal components and KeyTrainX
principalData = principalTwo * KeyTrainX.T


#Plotting principal component data
mpl.figure(2)
mpl.plot(principalData[0,:], principalData[1,:],"bo")
mpl.xlabel("Principal component 1")
mpl.ylabel("Principal component 2")
mpl.title("Principal two components scatterplot")
mpl.show()

#Assigns datapoints to one of two clusters 
def clusterAssign(dataset,centroid1,centroid2):
	cluster1 = NP.empty((0,21),float)
	cluster2 = NP.empty((0,21),float)
	for i in range(0,len(dataset)):
		if distance(centroid1, dataset[i]) < distance(centroid2, dataset[i]):
			cluster1 = NP.vstack((cluster1,dataset[i]))
		else:
			cluster2 = NP.vstack((cluster2,dataset[i]))

	return cluster1, cluster2

#Iterates cluster assigning and cetroid selection.
def twoMeansClustering(dataset,startCentroid1,startCentroid2):
	count = 0
	centroid1 = startCentroid1
	centroid2 = startCentroid2
	cluster1, cluster2 = clusterAssign(dataset, startCentroid1, startCentroid2)
	centroid1ajusted = NP.mean(cluster1,axis=0)
	centroid2ajusted = NP.mean(cluster2,axis=0)
	while distance(centroid1,centroid1ajusted) != 0.0 and distance(centroid2,centroid2ajusted) != 0.0:
		centroid1 = centroid1ajusted
		centroid2 = centroid2ajusted
		cluster1, cluster2 = clusterAssign(dataset, centroid1, centroid2)
		centroid1ajusted = NP.mean(cluster1,axis=0)
		centroid2ajusted = NP.mean(cluster2,axis=0)
		count += 1

	return cluster1, cluster2, centroid1, centroid2, count

#Computes distortion of clusters
def clusterDistortion(cluster,centroid):
	J = 0.0
	for i in range(0,len(cluster)):
		J += NP.power(distance(cluster[i], centroid),2)

	return J


#Computing, printing and plotting cluster, centroids and distortion
cluster1,cluster2, centroid1, centroid2, count = \
twoMeansClustering(KeyTrainX,KeyTrainX[166],KeyTrainX[402])

distortion1 = clusterDistortion(cluster1, centroid1)
distortion2 = clusterDistortion(cluster2, centroid2)

cluster1Prince = principalTwo * cluster1.T
cluster2Prince = principalTwo * cluster2.T

centroid1Prince = principalTwo * centroid1.T
centroid2Prince = principalTwo * centroid2.T

print ("Performance of 2-mean clustering:")
print (cluster1.shape)
print (cluster2.shape)
print (count)
print (distortion1)
print (distortion2)

#Plotting clusters
mpl.figure(3)
mpl.plot(cluster1Prince[0,:], cluster1Prince[1,:],"bo")
mpl.plot(cluster2Prince[0,:], cluster2Prince[1,:],"go")
mpl.plot(centroid1Prince[0,:], centroid1Prince[1,:],"ro")
mpl.plot(centroid2Prince[0,:], centroid2Prince[1,:],"ro")
mpl.plot()
mpl.title("2-means clustering scatterplot")
mpl.show()
