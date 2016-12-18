#Importing numpy and sqlite3
import numpy as NP
import sqlite3 as sql
from numpy.linalg import inv

#Printing title for solution txt-document
print ("# Output for datamining assigntment. Jonathan Hansen,fdg890 #")
print ("\n")

#Establishing connection to the database and creating cursor
conn = sql.connect('DataMiningAssignment2015.db')
c = conn.cursor()

#finding samplemean using sqlite3 library average function
c.execute("SELECT avg(target) FROM RedShift_Train_Y;")
for row in c:
	print ("Samplemean for Redshift_Train_Y:")
	print (row[0])
	print ("\n")

#finding biased sample variance using sqlite3 library average function
c.execute("SELECT avg((target - 0.1555464000000001)*\
	(target - 0.1555464000000001)) FROM RedShift_Train_Y;")

for row in c:
	print ("Biased sample variance for RedShift_Train_Y:")
	print (row[0])
	print ("\n")

#Importing 'target'-column from Redshift_Train_Y to an array
selectRedshift = c.execute ('SELECT target FROM Redshift_Train_Y')
getRow = c.fetchone ()

Redshift_Y_Array = [getRow]

for getRow in selectRedshift:
    Redshift_Y_Array.append (getRow)

Redshift_Y_Array = NP.matrix(Redshift_Y_Array)

#Importing data from Redshift_Train_X to an array
selectRedshift = c.execute ('SELECT * FROM Redshift_Train_X')
getRow = c.fetchone ()

Redshift_X_Array = [getRow]

for getRow in selectRedshift:
    Redshift_X_Array.append (getRow)

Redshift_X_Array = NP.matrix(Redshift_X_Array)

#Adding 1-column to Redshift_X_Array
Tilde_X = NP.insert (Redshift_X_Array, 10, 1, axis = 1)

#Calculating model parameters for Train
modelParameters = inv(Tilde_X.T*Tilde_X)*Tilde_X.T*Redshift_Y_Array

print ("Model parameters for Redshift_Train:")
print (modelParameters)
print ("\n")


#Calculating J for Train
J = 0
for i in range(0,2499):
    J = NP.power(Redshift_Y_Array[i]\
    	- NP.dot(modelParameters.T, Tilde_X[i].T),2) + J

#Calculating mean-sqaured error (MSE) for Train
MSE = J / 2500

print ("MSE for Redshift_Train")
print (MSE)
print ("\n")

#finding samplemean using sqlite3 library average function
c.execute("SELECT avg(target) FROM RedShift_Test_Y;")
for row in c:
	print ("Samplemean for Redshift_Test_Y:")
	print (row[0])
	print ("\n")

#finding biased sample variance using sqlite3 library average function
c.execute("SELECT avg((target - 0.15564839999999994)*\
	(target - 0.15564839999999994)) FROM RedShift_Test_Y;")

for row in c:
	print ("Biased sample variance for RedShift_Test_Y:")
	print (row[0])
	print ("\n")


#Importing 'target'-column from Redshift_Test_Y to an array
selectRedshift = c.execute ('SELECT target FROM Redshift_Test_Y')
getRow = c.fetchone ()

Redshift_Y_Array = [getRow]

for getRow in selectRedshift:
    Redshift_Y_Array.append (getRow)

Redshift_Y_Array = NP.matrix(Redshift_Y_Array)

#Importing data from Redshift_Test_X to an array
selectRedshift = c.execute ('SELECT * FROM Redshift_Test_X')
getRow = c.fetchone ()

Redshift_X_Array = [getRow]

for getRow in selectRedshift:
    Redshift_X_Array.append (getRow)

Redshift_X_Array = NP.matrix(Redshift_X_Array)

#Adding 1-column to Redshift_X_Array
Tilde_X = NP.insert (Redshift_X_Array, 10, 1, axis = 1)

#Calculating J for Test
J = 0
for i in range(0,2499):
    J = NP.power(Redshift_Y_Array[i]\
    	- NP.dot(modelParameters.T, Tilde_X[i].T),2) + J

#Calculating mean-sqaured error (MSE) for Test
MSE = J / 2500

print ("MSE for Redshift_Test:")
print (MSE)
print ("\n")
