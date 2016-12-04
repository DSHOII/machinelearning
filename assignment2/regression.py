import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl


#__________________________Linear regression model_____________________________#

# Location variable for data
locationData = 'DanWood.dt'

# Reading data into dataframes
dfData = pd.read_csv(locationData, header=None, sep=' ')

# Extracting X and Y from training and test data into matrices
inputX = dfData.as_matrix(columns=[0])
targetY = dfData.as_matrix(columns=[1])

# Adding 1-columns to input matrix
ones = np.ones((len(inputX),1), dtype=np.int)

bigX = np.hstack((inputX, ones))

# # Plotting data just to have a look at it
# mpl.figure(1)
# mpl.plot(inputX, targetY,"bo")
# mpl.xlim([0,2])
# mpl.ylim([0,6])
# mpl.xlabel("Absolute temperature")
# mpl.ylabel("Radiated energy")
# mpl.title("Radiated energy from carbon filament")
# mpl.show()

# Transposing input matrix
inputT = np.transpose(bigX)

# Transposing target
targetT = np.transpose(targetY)

# Dotproduct of input and transposed input
dotProductInput = np.dot(inputT, bigX)

# Dotproduct of transposed target and transposed input
dotProductTarget = np.dot(inputT, targetY)

# Inverse of dot product
inverse = np.linalg.inv(dotProductInput)

# Dotproduct of the whole damn thing, thas is our w and b
dotProductFinal = np.dot(inverse, dotProductTarget)

# Linear regression


print(dotProductFinal)
