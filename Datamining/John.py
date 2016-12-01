import sqlite3
import math
import numpy as np
import numpy.linalg as npl
import matplotlib
import pylab

#Defining database connection and cursors
conn = sqlite3.connect('DataMiningAssignment2015.db')
KsTr_X = conn.execute("\
	SELECT *\
	FROM Keystrokes_Train_X;\
	")

KsTr_Y = conn.execute("\
	SELECT *\
	FROM Keystrokes_Train_Y;\
	")

KsTe_X = conn.execute("\
	SELECT *\
	FROM Keystrokes_Test_X;\
	")

KsTe_Y = conn.execute("\
	SELECT *\
	FROM Keystrokes_Test_Y;\
	")

trainX = np.array(KsTr_X.fetchall())
trainY = np.array(KsTr_Y.fetchall())
testX = np.array(KsTe_X.fetchall())
testY = np.array(KsTe_Y.fetchall())

covMat = np.cov(trainX.T)
eigVal, eigVec = npl.eigh(covMat)

print (eigVal.shape)
print (eigVec.shape)


#Function to find the least amount of components, to explain a specific percentage of the variance of a dataset.
def noCompExplainPercentage(trainX, explPerc):
	covMat = np.cov(trainX.T)
	eigVal, eigVec = npl.eigh(covMat)
	#Mirrors eigenvalue array, so eigenvalues appear in decreasing order
	eigVal = eigVal[::-1]
	noOfComp = 0
	eigValCompSum = 0.0
	eigValSum =  np.sum(eigVal)
	for i in range(0,len(eigVal)):
		if (eigValCompSum / eigValSum) * 100 < explPerc:
			eigValCompSum += eigVal[noOfComp]
			noOfComp += 1
		else:
			explVar = (eigValCompSum / eigValSum) * 100
	return ("number of components:", noOfComp, "explained variance:", explVar)

print(noCompExplainPercentage(trainX, 90))


'''

john = NP.cov(KeyTrainX.T)

karsten, knud = NP.linalg.eig(john)

mpl.plot(0.7,0.7,"bo")
mpl.show()

print (john.shape)
print (karsten.shape)
print (knud.shape)

'''