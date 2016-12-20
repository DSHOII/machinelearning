import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl


#___________________________Principal Componet Analysis________________________#

# Location variable for training data.
locationData = 'ML2016TrafficSignsTrain.csv'

# Reading data into dataframe.
dfData = pd.read_csv(locationData, header=None, sep=',')

# Extracting data into matrices of input and labels.
trafficSignData = dfData.as_matrix()

shape = trafficSignData.shape

trafficSignInput = trafficSignData[:,0:(shape[1]-1)]
trafficSignLabels = trafficSignData[:,shape[1]-1].astype(int)


# Creating vectors for plotting.

labelCount = np.bincount(trafficSignLabels)
labelFreq = np.divide(labelCount, shape[0])

labelXaxis = np.arange(1,44)


# Plotting histogram of distribution of class frequencies.
mpl.figure(1)
mpl.hist(trafficSignLabels, bins=43, normed=1)
mpl.title("Label frequency histogram")
mpl.xlabel("Label")
mpl.ylabel("Frequency")
mpl.show()

# Computing empirical covariance matrix.
trafficSignInputCov = np.cov(trafficSignInput.T)

# computing eigen values and eigen vectors of the covariance matrix.
eigenVal, eigenVec = np.linalg.eigh(trafficSignInputCov)


# Creating vector to go as X-axis in scatter plot.
trafficSignXaxis = np.arange(1,1569)

# Reordering the eigenvalues in decreasing order.
eigenValOrder = eigenVal[::-1]


# Plotting the eigenspectrum.
mpl.figure(2)
mpl.plot(trafficSignXaxis, eigenValOrder, "bo")
mpl.xlim([0,1570])
mpl.ylim([0,0.33])
mpl.xlabel("Eigenvalue number")
mpl.ylabel("Eigenvalue")
mpl.title("Eigenspectrum")
mpl.show()


# Function that sums a specified number of values in a matrix from top down.
def matrixTopSum(eigenValues, topCount):
	sumCount = 0
	eigenSum = 0.0
	for i in range(0,topCount):
		if sumCount < topCount:
			eigenSum += np.sum(eigenValues[sumCount])
			sumCount += 1

	return eigenSum



#Computing total sum of eigen values
eigenValTotalSum = np.sum(eigenVal)


#PCA model computing the explained variance from a number of components.
def PCA(eigenValues, desiredExplainedPrc):
	compCount = 0
	explainedVar = 0.0
	for i in range(0,len(eigenValues)):
		if matrixTopSum(eigenValues,compCount) / eigenValTotalSum * 100 < desiredExplainedPrc:
			explainedVar = matrixTopSum(eigenValues,compCount+1) / eigenValTotalSum * 100
			compCount += 1
		else:
			compCount

	return compCount, explainedVar

print ('PCA model: Number of components needed and variance explained:')
print (PCA(eigenValOrder,90))
print ('\n')


# Creating matrix of first two principal components.
principalTwo = np.vstack((eigenVec[:,-1], eigenVec[:,-2]))

# Computing data shifted by two principal components.
principalData = np.dot(principalTwo, trafficSignInput.T)

# Creating new labels according to shape.
def relabel(oldLabels):
    newLabels = np.arange(len(oldLabels), dtype=(float,3))
    roundS = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
    upTri = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    diamond = [12]
    downTri = [13]
    octagon = [14]
    for x in range(0,len(oldLabels)):
        if oldLabels[x] in roundS:
            newLabels[x] = (0.0, 0.0, 1.0)
        elif oldLabels[x] in upTri:
            newLabels[x] = (1.0, 0.0, 0.0)
        elif oldLabels[x] in diamond:
            newLabels[x] = (0.0, 0.5, 0.0)
        elif oldLabels[x] in downTri:
            newLabels[x] = (0.75, 0.75, 0)
        else:
            newLabels[x] = (0.75, 0, 0.75)
    return newLabels


trafficSignColor = relabel(trafficSignLabels)

# Plotting the data shifted by the two principals components using different
# colors for different shapes.
mpl.figure(3)
mpl.scatter(principalData[0,:], principalData[1,:], c=trafficSignColor)
mpl.show()

# Bringing in distance function from last weeks assignment.
def distance(vector1, vector2):
	return np.linalg.norm(vector1 - vector2)



# Function that calculates and returns the minimum distance from point to 4
# other points.
def minD4P(datapoint, p1, p2, p3, p4):
    distances = np.empty(4)
    distances[0] = distance(datapoint, p1)
    distances[1] = distance(datapoint, p2)
    distances[2] = distance(datapoint, p3)
    distances[3] = distance(datapoint, p4)
    mindistance = np.amin(distances)
    return mindistance

# Function for assigning data points to 1 of 4 clusters with given centroids.
def clusterAssign(dataSet, c1, c2, c3, c4):
    cluster1 = np.empty([0,np.size(dataSet,1)], float)
    cluster2 = np.empty([0,np.size(dataSet,1)], float)
    cluster3 = np.empty([0,np.size(dataSet,1)], float)
    cluster4 = np.empty([0,np.size(dataSet,1)], float)
    for i in range(0,len(dataSet)):
        if distance(c1, dataSet[i]) == minD4P(dataSet[i], c1, c2, c3, c4):
            cluster1 = np.vstack((cluster1,dataSet[i]))
        elif distance(c2, dataSet[i]) == minD4P(dataSet[i], c1, c2, c3, c4):
            cluster2 = np.vstack((cluster2,dataSet[i]))
        elif distance(c3, dataSet[i]) == minD4P(dataSet[i], c1, c2, c3, c4):
            cluster3 = np.vstack((cluster3,dataSet[i]))
        else:
            cluster4 = np.vstack((cluster4,dataSet[i]))
    return cluster1, cluster2, cluster3, cluster4


# Function that iterates cluster assigning and centroid selection.
def fourMeansClustering(dataSet, startC1, startC2, startC3, startC4):
    count = 0
    c1 = startC1
    c2 = startC2
    c3 = startC3
    c4 = startC4
    cluster1, cluster2, cluster3, cluster4 = clusterAssign(dataSet, startC1, startC2, startC3, startC4)
    c1New = np.mean(cluster1, axis=0)
    c2New = np.mean(cluster2, axis=0)
    c3New = np.mean(cluster3, axis=0)
    c4New = np.mean(cluster4, axis=0)
    while distance(c1, c1New) != 0.0 and distance(c2, c2New) != 0.0 and distance(c3, c3New) and distance(4, c4New):
        c1 = c1New
        c2 = c2New
        c3 = c3New
        c4 = c4New
        cluster1, cluster2, cluster3, cluster4 = clusterAssign(dataSet, c1, c2, c3, c4)
        c1New = np.mean(cluster1,axis=0)
        c2New = np.mean(cluster2,axis=0)
        c3New = np.mean(cluster3,axis=0)
        c4New = np.mean(cluster4,axis=0)
        count += 1
    return cluster1, cluster2, cluster3, cluster4, c1, c2, c3, c4, count


# Performing 4-means clustering on the data and extracting results.
res = fourMeansClustering(trafficSignInput, trafficSignInput[0], trafficSignInput[1], trafficSignInput[2], trafficSignInput[3])

cluster1 = res[0]
cluster1 = res[1]
cluster1 = res[2]
cluster1 = res[3]
clusterCenter1 = res[4]
clusterCenter2 = res[5]
clusterCenter3 = res[6]
clusterCenter4 = res[7]
numberOfIterations = res[8]

# Combining cluster centers to two new matrices
clusterCenters1 = np.vstack((clusterCenter1, clusterCenter2))
clusterCenters2 = np.vstack((clusterCenter3, clusterCenter4))

# Shifting cluster centers.
cc1Shifted = np.dot(clusterCenters1, principalTwo.T)
cc2Shifted = np.dot(clusterCenters2, principalTwo.T)

ccShifted = np.hstack((cc1Shifted, cc2Shifted))

# Making scatterplot as above, but with the cluster centers added.
x = range(-4,4)
y = range(-4,3)
fig = mpl.figure
axis = fig().add_subplot(111)

axis.scatter(principalData[0,:], principalData[1,:], c=trafficSignColor, label='Data points')
axis.scatter(ccShifted[0,:], ccShifted[1,:], c='c', s=50, label='Cluster centers', marker="s")
mpl.legend(loc='lower left');
mpl.show()

######### End of script ################################
