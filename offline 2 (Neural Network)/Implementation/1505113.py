import  numpy as np
import random
maximumIteration = 1


#---------------------------------------------------------------------------------------
#reading the trainning dataset
f = open("./trainNN.txt", "r")

#reading the lines of the dataset
lines = f.readlines()
f.close()

numOfFeatures = len(lines[0].split()) - 1

#Counter for counting number of classes
numOfClass = 0

#declaring vectors for Features and Classes
featureVector = []
classVector = []

for line in lines:
    lineData = line.split()

    className = int(lineData[numOfFeatures])
    classVector.append(className)

    tempVector =  (lineData[:numOfFeatures])

    #converting the feature vales to flaot
    for i in range(numOfFeatures):
        tempVector[i] = float (tempVector[i])

    featureVector.append(tempVector)

    if(className > numOfClass):
        numOfClass = className

#converitng feature vector and class vector to numpy array.
trainningX = np.array(featureVector).T
trainningY = np.array(classVector)

#printing the class and feature informations
print('No of features: ', numOfFeatures)
print('No of Class: ', numOfClass)
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#scaling the trainning dataset

#calculating the feature vectors mean and standart deviation first
#axis=1 means the mean is calculated along the row of the feature matrix
tempMean = trainningX.mean(axis=1)
#converting the mean row vector to a column vector
mean = tempMean.reshape(numOfFeatures,1)

#axis=1 means standard deviation is calculated along the row of the feature matrix
tempSd = trainningX.std(axis=1)
#converting the standard deviation row vector to a column vector
sd = tempSd.reshape(numOfFeatures,1)

#normalizing rule :  (feature vector - mean vector)/ standard deviation vector
trainningX = (trainningX - mean)/sd
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#defining number of Hidden layers. Output layer is included in the hidden layer
numOfHiddenLayers = 3

#defining the number of nodes in the hidden layers. Output layer number of nodes = number of class
numOfNodesHiddenLayer = 3

#creating an array to keep track which layer has how many nodes
perLayerNodesNumber = []

perLayerNodesNumber.append(numOfFeatures)

for i in range(numOfHiddenLayers-1):
    perLayerNodesNumber.append(numOfNodesHiddenLayer)

perLayerNodesNumber.append(numOfClass)
print('Per layer node numbers: ',perLayerNodesNumber)

#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#Creating the weight matrix
weightMatrix = []
weightMatrix.append(0.0)

#creating the matrix from hidden layer 1. Input layer does not need weight matrix
for i in range(1, numOfHiddenLayers+1 , 1 ):

    thisLayerNodeNumber = perLayerNodesNumber[i]
    previousLayerNodeNumber = perLayerNodesNumber[i-1]

    thisLayerWeightMatrix = []

    thisLayerWeightMatrix = [[ 0.0 for i in range(previousLayerNodeNumber) ] for j in range(thisLayerNodeNumber)]

    for j in range(thisLayerNodeNumber):
        for k in range(previousLayerNodeNumber):
            thisLayerWeightMatrix[j][k] = random.random()

    thisLayerWeightMatrix = np.array(thisLayerWeightMatrix)

    weightMatrix.append(thisLayerWeightMatrix)

weightMatrix = np.array(weightMatrix)
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#working of Forward Propagation and Backward Propagation

#sigmoid function. It returns a vector
def sigmoid(vector):
    return (1 / (1 + np.exp(-vector)))


for i in range(maximumIteration):

    #working of forward propagation
    Yvector = np.array(trainningX)
    for j in range(1, numOfHiddenLayers+1 , 1):
        mul = np.dot(weightMatrix[j] , Yvector )
        Yvector = sigmoid(mul)

    Ycap = np.array(Yvector)





































