import  numpy as np

#reading the trainning dataset

f = open("./trainNN.txt", "r")

#reading the lines of the dataset
lines = f.readlines()
f.close()

noOfFeatures = len(lines[0].split()) - 1

#Counter for counting number of classes
noOfClass = 0

#declaring vectors for Features and Classes
featureVector = []
classVector = []

for line in lines:
    lineData = line.split()

    className = int(lineData[noOfFeatures])
    classVector.append(className)

    tempVector =  (lineData[:noOfFeatures])

    #converting the feature vales to flaot
    for i in range(noOfFeatures):
        tempVector[i] = float (tempVector[i])

    featureVector.append(tempVector)

    if(className > noOfClass):
        noOfClass = className

#converitng feature vector and class vector to numpy array.
trainningX = np.array(featureVector).T
trainningY = np.array(classVector)

#printing the class and feature informations
print('No of features: ', noOfFeatures)
print('No of Class: ', noOfClass)

print(trainningX.shape)
print(trainningY.shape)
