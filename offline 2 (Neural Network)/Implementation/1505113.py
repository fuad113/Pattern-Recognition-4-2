import  numpy as np
import copy

#maximum iteration number
maximumIteration = 1000

np.random.seed(1)

#learning rate
mu= 0.01

#---------------------------------------------------------------------------------------
#reading the trainning dataset
f = open("./trainNN.txt", "r")

#reading the lines of the dataset
lines = f.readlines()
f.close()

numOfFeatures = len(lines[0].split()) - 1

numOfsamples = len(lines)


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

#Trainnig Y is kept for tracking which sample is classified as which class. its row= num of class + 1 (avoid O indexing)
#its column = num of samples
trainningY = np.zeros((numOfClass , numOfsamples))

for i in range(numOfsamples):
    trainningY[classVector[i]-1][i] =1

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
numOfLayers = 4

#defining the number of nodes in the hidden layers. Output layer number of nodes = number of class
numOfNodesPerLayer = 4

#creating an array to keep track which layer has how many nodes
perLayerNodesNumber = []

perLayerNodesNumber.append(numOfFeatures)

for i in range(numOfLayers - 1):
    perLayerNodesNumber.append(numOfNodesPerLayer)

perLayerNodesNumber.append(numOfClass)
print('Per layer node number: ',perLayerNodesNumber)

#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#Creating the weight matrix
weightMatrix = []
weightMatrix.append(0.0)

#creating the matrix from hidden layer 1. Input layer does not need weight matrix
for i in range(1, numOfLayers + 1 , 1):

    thisLayerNodeNumber = perLayerNodesNumber[i]
    previousLayerNodeNumber = perLayerNodesNumber[i-1]

    thisLayerWeightMatrix = []

    thisLayerWeightMatrix = [[ 0.0 for i in range(previousLayerNodeNumber) ] for j in range(thisLayerNodeNumber)]

    for j in range(thisLayerNodeNumber):
        for k in range(previousLayerNodeNumber):
            thisLayerWeightMatrix[j][k] = np.random.randn()

    thisLayerWeightMatrix = np.array(thisLayerWeightMatrix)

    weightMatrix.append(thisLayerWeightMatrix)

weightMatrix = np.array(weightMatrix)
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#working of Forward Propagation

#3d vectors/array to save the values of Y and V
yDic = {}
vDic = {}

#sigmoid function. It returns a vector
def sigmoid(vector):
    return (1 / (1 + np.exp(-vector)))

#sigmoid differentiated function
def sigmoidDifferentiated(vector):
    return (sigmoid(vector) * (1 - sigmoid(vector)))

def forwardPropagation(featureVec):
    global yDic,vDic

    y = np.array(featureVec)
    yDic[str(0)] = y
    for i in range(1, numOfLayers + 1 , 1):
        v = np.dot(weightMatrix[i] , y)
        y= sigmoid(v)
        yDic[str(i)] = y
        vDic[str(i)] = v
    return y
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#working of backword Propagation
#calculation of delta rj
def delRJCalculation(sampleNum, layerNum, previousLayerDelRJ):

    if(layerNum == numOfLayers):
        value1 = yDic[str(layerNum)][:,sampleNum]
        value2 = trainningY[:,sampleNum]

        tempRJ = np.subtract(value1 , value2)

        value3 = vDic[str(layerNum)][:,sampleNum]
        fPrime = sigmoidDifferentiated(value3)

        delRJ = np.multiply(tempRJ, fPrime)

    else:
        value1 = vDic[str(layerNum)][:,sampleNum]
        fPrime = sigmoidDifferentiated(value1)

        fPrimeShape = fPrime.shape
        fPrime = fPrime.reshape(fPrimeShape[0] , 1)

        value2 = weightMatrix[layerNum + 1]
        value3 = np.diagflat(fPrime)
        tempRJ = np.dot(value2 , value3)

        value4 = previousLayerDelRJ.T
        value5 = tempRJ
        delRJ = np.dot(value4,value5).T

    return delRJ

def backwardPropagation():
    weightMatrixNew = copy.deepcopy(weightMatrix)

    for i in range(numOfsamples):
        delRJ = None
        for j in range(numOfLayers , 0  , -1):
            delRJ  = delRJCalculation(i , j , delRJ)
            #reshape delRj to make it a column matrix
            delRJShape = delRJ.shape
            delRJ = delRJ.reshape(delRJShape[0] , 1)

            #size of previous Y
            prevYsize = len(yDic[str(j-1)][:,i])

            #multiplication part and updating part
            value1 = delRJ
            value2 = yDic[str(j-1)][:,i].reshape(1, prevYsize)
            dotProdValue = np.dot(value1, value2)

            tempW= mu * dotProdValue

            weightMatrixNew[j] = weightMatrixNew[j] - tempW

    for i in range(1 , numOfLayers + 1 , 1):
        weightMatrix[i] = weightMatrixNew[i]

#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#trainning the dataset
print("Trainning On Process")
for i in range(maximumIteration):
    yCap =forwardPropagation(trainningX)
    backwardPropagation()
#---------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------
#reading the testing dataset
f = open("./testNN.txt", "r")

#reading the lines of the dataset
lines = f.readlines()
f.close()

numOfsamplesTest = len(lines)

#declaring vectors for Features and Classes
featureVectorTest = []
classVectorTest = []

for line in lines:
    lineData = line.split()

    className = int(lineData[numOfFeatures])
    classVectorTest.append(className)

    tempVector =  (lineData[:numOfFeatures])

    #converting the feature vales to flaot
    for i in range(numOfFeatures):
        tempVector[i] = float (tempVector[i])

    featureVectorTest.append(tempVector)


#converitng feature vector and class vector to numpy array.
testingX = np.array(featureVectorTest).T
testingY = copy.deepcopy(classVectorTest)
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
#scaling the testing dataset

#calculating the feature vectors mean and standart deviation first
#axis=1 means the mean is calculated along the row of the feature matrix
tempMean = testingX.mean(axis=1)
#converting the mean row vector to a column vector
mean = tempMean.reshape(numOfFeatures,1)

#axis=1 means standard deviation is calculated along the row of the feature matrix
tempSd = testingX.std(axis=1)
#converting the standard deviation row vector to a column vector
sd = tempSd.reshape(numOfFeatures,1)

#normalizing rule :  (feature vector - mean vector)/ standard deviation vector
testingX = (testingX - mean)/sd
#---------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------
#testing work
print("Testing On Process")

report = open("NN Misclassified Report.txt", "w")
report.write("Neural Network Misclassified Report\n")
report.write("Misclassified Samples:\n\n")

report.write(
    "Feature No.                            Feature Values                            Actual Class           Predicted Class\n")


correctClassified = 0

yCap = forwardPropagation(testingX)

for i in range(numOfsamplesTest):
    min = yCap[0][i]
    predictedClass = 1

    for j in range(numOfFeatures):
        if(yCap[j][i] > min ):
            min =yCap[j][i]
            predictedClass = j+1

    actualClass = testingY[i]

    if(actualClass == predictedClass):
        correctClassified += 1
    else:
        report.write(str(i + 1) + "              " + str(testingX[:,i]) + "                        "
        + str(actualClass) + "                     " + str(predictedClass) + "\n")


accuracy = (correctClassified / numOfsamplesTest) * 100

print('Accuracy: ' , accuracy ,'%')