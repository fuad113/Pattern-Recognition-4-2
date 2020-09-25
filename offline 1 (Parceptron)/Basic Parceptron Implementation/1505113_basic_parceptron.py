import numpy as np
import math

f = open("./trainLinearlySeparable.txt", "r")
first_line = f.readline()

temp = first_line.split()

no_of_features = int(temp[0])
no_of_class = int(temp[1])
no_of_samples = int(temp[2])

# reading files except the 1st line and filling the dictionary
# dictionary contains the number of classes
lines = f.readlines()[0:]
f.close()

# training of the dataset

print("Parceptron Trainning")

#defining the value of rho
rho=0.1
#creating the W(t) matrix
weightMatrix=[]

for i in range(no_of_features+1):
    weightMatrix.append(0.0)

weightMatrix =np.array(weightMatrix)

# np.random.seed(113)
#
# weightMatrix = np.random.uniform(-1, 1, no_of_features + 1)

for i in range(1000):

    #defining the misclassified feature vector set
    misClassified = []

    for j in range(no_of_samples):

        lineData = lines[j].split()

        className = lineData[no_of_features]

        className =int(className)

        #creating feature vector
        featureVector= lineData[:no_of_features]

        # append 1 in the end of the feature vector
        featureVector.append(1.0)

        #converting the values of feature vector to float
        for k in range(no_of_features+1):
            featureVector[k]= float(featureVector[k])

        featureVector = np.array(featureVector)

        dotValue = np.dot(weightMatrix, featureVector)

        if(dotValue >= 0 ):
            experimentalClass = 1
        else:
            experimentalClass = 2

        #checking if misclassified and multiplying delX with feature vector
        if(experimentalClass== 1 and className== 2):
            delX= 1
            featureVector = featureVector * delX
            misClassified.append(featureVector)

        elif(experimentalClass == 2 and className == 1):
            delX = -1
            featureVector = featureVector * delX
            misClassified.append(featureVector)

    #checking if the misclassified array is zero or not. If zero then converged. Break the loop
    if(len(misClassified) == 0):
        break

    #updating the Weight vector
    sumVector =[]
    for k in range(no_of_features+1):
        sumVector.append(0.0)

    sumVector = np.array(sumVector)

    for k in range(len(misClassified)):
        sumVector+=misClassified[k]*rho

    weightMatrix = weightMatrix - sumVector


#testing of data set

print("Parceptron Testing")

f2 = open("./testLinearlySeparable.txt", "r")
lines = f2.readlines()[0:]
f2.close()

correctClassifyCounter = 0

for i in range(no_of_samples):
    lineData = lines[i].split()

    className = lineData[no_of_features]

    xVector = lineData[:no_of_features]

    xVector.append(1.0)

    for k in range(no_of_features + 1):
        xVector[k] = float(xVector[k])

    xVector =np.array(xVector)


    dotProdValue = np.dot(weightMatrix,xVector)

    #checking the value of w*X. if w*x > 0 then class 1 otherwise class 2
    if dotProdValue > 0:
        predictedClass = 1
    else:
        predictedClass = 2

    className =int(className)
    predictedClass = int(predictedClass)

    if (className == predictedClass) :
        correctClassifyCounter = correctClassifyCounter + 1


print("CorrectClassify: ", correctClassifyCounter)

accuracy = (correctClassifyCounter / no_of_samples) * 100

print("Accuracy: ",accuracy, "%")
