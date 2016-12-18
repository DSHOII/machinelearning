import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl


#__________________________Linear regression model_____________________________#
# Linear aggression model. Assumes data in matrices. Since data might be given
# in the wrong shapes this is the first thing that we check. This check assumes
# more data points than number of dimensions on vector. I make substantial use
# of the library Numpy to do matrix operations.


# Making sure target is in the right dimensions.
def target(outputData):
    if outputData.shape[0] > outputData.shape[1]:
        targetY = outputData
    else:
        targetY = np.transpose(outputData)
    return targetY

# Preparing input, X tilde.
def bigX(inputData):
    if inputData.shape[0] > inputData.shape[1]:
        inputNoTilde = inputData
    else:
        inputNoTilde = np.transpose(inputData)
    ones = np.ones((len(inputNoTilde), 1), dtype=np.int)
    inputX = np.hstack((inputNoTilde, ones))
    return inputX

# Combining dot product and inverse into one function.
def dotInverse(someX):
    someXTrans = np.transpose(someX)
    dotProduct = np.dot(someXTrans, someX)
    result = np.linalg.inv(dotProduct)
    return result

# The actual linear regression model.
def linearReg(inputDataX, inputDataY):
    targetY = target(inputDataY)
    Xtilde = bigX(inputDataX)
    XtildeTrans = np.transpose(Xtilde)
    XtildeTransDotInv = dotInverse(Xtilde)
    affLinPreModel = np.dot(XtildeTransDotInv,np.dot(XtildeTrans, targetY))
    return (affLinPreModel)



#_____________________Affine linear model of DanWood.dt________________________#


# Location variable for data.
locationData = 'DanWood.dt'

# Reading data into dataframes.
dfData = pd.read_csv(locationData, header=None, sep=' ')

# Extracting X and Y from training and test data into matrices.
inputX = dfData.as_matrix(columns=[0])
inputY = dfData.as_matrix(columns=[1])

# Building affine linear regression model and reporting results.
vectorWandB = linearReg(inputX, inputY)

vectorW = np.transpose(vectorWandB[:-1])

b = vectorWandB[-1]

print(vectorWandB)
print('\n')
print('Affine linear model build. w: ' + str(vectorW) + '   b: '
                                             + str(b))



# Empirical Y computer.
def empiricalY(x, vectorW, b):
    yHat = np.dot(np.transpose(vectorW), x) + b
    return yHat

# Squared loss computer
def squaredLoss(y, yHat):
    loss = y - yHat
    squaredLoss = loss * loss
    return squaredLoss

# Mean-squared loss computer
def msq(inputX, inputY, w, b):
    n = len(inputY)
    loss = 0
    for i in range(0, len(inputX)):
        yHat = empiricalY(inputX[i], w, b)
        loss += squaredLoss(inputY[i], yHat)
    return np.divide(loss, n)


# Reporting mean-squared error.
meanSquaredError = msq(inputX, inputY, vectorW, b)

print('Mean-squared-error of the model is: ' + str(meanSquaredError))


# Plotting data and regression line. Regression line by two points.
regPointX = [0,10]
regPointY = [b[0],empiricalY(10, vectorW,b)[0,0]]

mpl.figure(1)
mpl.plot(inputX, inputY,"bo")
mpl.plot(regPointX, regPointY, "bo", linestyle='-', color='r')
mpl.xlim([0,2])
mpl.ylim([0,6])
mpl.xlabel("Absolute temperature")
mpl.ylabel("Radiated energy")
mpl.title("Radiated energy from carbon filament")
mpl.show()


#### End of script ####
