import numpy as np
import pandas as pd
# import matplotlib.pyplot as mpl


#_________________________Logistic regression model____________________________#

# Location variable for training and test data
locationTrain = 'IrisTrainML.dt'
locationTest = 'IrisTestML.dt'

# Reading data into dataframes.
dfTrain = pd.read_csv(locationTrain, header=None, sep=' ')
dfTest = pd.read_csv(locationTest, header=None, sep=' ')

# Preparing data: Filter our class 2.
dfTrainNO2 = dfTrain.loc[dfTrain[2] != 2]
dfTestNO2 = dfTest.loc[dfTest[2] != 2]


# Extracting X and Y from training and test data into matrices.
irisTrainX = dfTrainNO2.as_matrix(columns=[0,1])
irisTrainY = dfTrainNO2.as_matrix(columns=[2])
irisTestX = dfTestNO2.as_matrix(columns=[0,1])
irisTestY = dfTestNO2.as_matrix(columns=[2])

# Replacing class 0 with -1
np.place(irisTrainY, irisTrainY == 0, -1)
np.place(irisTestY, irisTestY == 0, -1)


# Preparing input, X tilde.
def bigX(inputData):
    if inputData.shape[0] > inputData.shape[1]:
        inputNoTilde = inputData
    else:
        inputNoTilde = np.transpose(inputData)
    ones = np.ones((len(inputNoTilde), 1), dtype=np.int)
    inputX = np.hstack((inputNoTilde, ones))
    return inputX


# Function that creates weights
def initWeights(data):
    dimensions = data.shape[1]
    weights = np.zeros((dimensions))
    return weights


# Function for computing a single gradient.
def singleGradient(x, y, w):
    yMULTx = np.multiply(y,x)
    wDOTx = np.dot(np.transpose(w),x)
    yMULTwtx = np.multiply(y, wDOTx)
    exp = np.exp(yMULTwtx)
    gradient = np.divide(yMULTx, 1 + exp)
    return gradient


# Function that computes full gradient.
def gradient(dataX, dataY, weights):
    accumulator = initWeights(dataX)
    n = len(dataY)
    for i in range(len(dataX)):
        gradient = singleGradient(dataX[i], dataY[i], weights)
        accumulator += gradient
    mean = np.divide(accumulator, n)
    gradient = np.negative(mean)
    return gradient


# Function for updating weights.
def updateWeights(oldWeights, direction, learningRate):
    newWeight = oldWeights + np.multiply(learningRate, direction)
    return newWeight

# Logistic regression model. Implementation of LFD algorithm.
def logReg(dataX, dataY, learningRate):
    X = bigX(dataX)
    weights = initWeights(X)
    for i in range(0,1000):
        g = gradient(X, dataY, weights)
        direction = np.negative(g)
        weights = updateWeights(weights, direction, learningRate)
    return weights


# Building affine linear model and reporting results.
vectorWandB = logReg(irisTrainX, irisTrainY, 0.1)

vectorW = np.transpose(vectorWandB[:-1])

b = vectorWandB[-1]

print(vectorWandB)
print('\n')
print('Affine linear model build. w: ' + str(vectorW) + '   b: '
                                             + str(b))



