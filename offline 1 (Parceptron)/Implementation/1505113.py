import numpy as np

maxIteration =1000

def BasicParceptron():
    f = open("./datasets/trainLinearlySeparable.txt", "r")
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

    # defining the value of rho
    rho = 0.1
    # creating the W(t) matrix
    weightVector = []

    for i in range(no_of_features + 1):
        weightVector.append(0.0)

    weightVector = np.array(weightVector)

    for i in range(maxIteration):

        # defining the misclassified feature vector set
        misClassified = []

        for j in range(no_of_samples):

            lineData = lines[j].split()

            className = lineData[no_of_features]

            className = int(className)

            # creating feature vector
            featureVector = lineData[:no_of_features]

            # append 1 in the end of the feature vector
            featureVector.append(1.0)

            # converting the values of feature vector to float
            for k in range(no_of_features + 1):
                featureVector[k] = float(featureVector[k])

            featureVector = np.array(featureVector)

            dotValue = np.dot(weightVector, featureVector)

            if (dotValue >= 0):
                experimentalClass = 1
            else:
                experimentalClass = 2

            # checking if misclassified and multiplying delX with feature vector
            if (experimentalClass == 1 and className == 2):
                delX = 1
                featureVector = featureVector * delX
                misClassified.append(featureVector)

            elif (experimentalClass == 2 and className == 1):
                delX = -1
                featureVector = featureVector * delX
                misClassified.append(featureVector)

        # checking if the misclassified array is zero or not. If zero then converged. Break the loop
        if (len(misClassified) == 0):
            break

        # updating the Weight vector
        sumVector = []
        for k in range(no_of_features + 1):
            sumVector.append(0.0)

        sumVector = np.array(sumVector)

        for k in range(len(misClassified)):
            sumVector += misClassified[k] * rho

        weightVector = weightVector - sumVector

    # testing of data set
    print("Parceptron Testing")

    f2 = open("./datasets/testLinearlySeparable.txt", "r")

    report = open("Parceptron Report.txt","w")
    report.write("Basic Parceptron Report\n")
    report.write("Misclassified Samples:\n\n")

    report.write(
        "Feature No.                            Feature Values                            Actual Class           Predicted Class\n")

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

        xVector = np.array(xVector)

        dotProdValue = np.dot(weightVector, xVector)

        # checking the value of w*X. if w*x > 0 then class 1 otherwise class 2
        if dotProdValue > 0:
            predictedClass = 1
        else:
            predictedClass = 2

        className = int(className)
        predictedClass = int(predictedClass)

        if (className == predictedClass):
            correctClassifyCounter = correctClassifyCounter + 1
        else:
            report.write(str(i + 1) + "              " + str(lineData[:no_of_features]) + "             "
                         + str(className) + "                     " + str(predictedClass) + "\n")


    report.write("\n")
    print("Parceptron CorrectClassify: ", correctClassifyCounter)

    accuracy = (correctClassifyCounter / no_of_samples) * 100

    print("Parceptron Accuracy: ", accuracy, "%\n")
    report.write("Accuracy: " + str(accuracy) + "%\n")


def RewardAndPunishment():
    f = open("./datasets/trainLinearlySeparable.txt", "r")
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
    print("Reward and Punishment Trainning")

    # defining the value of rho
    rho = 0.1
    # creating the W(t) matrix
    weightVector = []

    for i in range(no_of_features + 1):
        weightVector.append(0.0)

    weightVector = np.array(weightVector)

    for i in range(maxIteration):

        #taking a counter to count the misClassified samples
        misClassifiedCounter = 0

        for j in range(no_of_samples):

            lineData = lines[j].split()

            className = lineData[no_of_features]

            className = int(className)

            # creating feature vector
            featureVector = lineData[:no_of_features]

            # append 1 in the end of the feature vector
            featureVector.append(1.0)

            # converting the values of feature vector to float
            for k in range(no_of_features + 1):
                featureVector[k] = float(featureVector[k])

            featureVector = np.array(featureVector)

            dotValue = np.dot(weightVector, featureVector)

            if (dotValue >= 0):
                experimentalClass = 1
            else:
                experimentalClass = 2

            # checking if misclassified and multiplying delX with feature vector
            if (experimentalClass == 2 and className == 1):
                featureVector =featureVector * rho
                weightVector = weightVector + featureVector
                misClassifiedCounter+=1

            elif (experimentalClass == 1 and className == 2):
                featureVector = featureVector*rho
                weightVector = weightVector - featureVector
                misClassifiedCounter+=1

        # converges when no misclassified samples then break the iteration.
        if(misClassifiedCounter == 0):
            break

    # testing of data set

    print("Reward and Punishment Testing")

    f2 = open("./datasets/testLinearlySeparable.txt", "r")

    report = open("Reward and Punishment Report.txt","w")
    report.write("Reward and Punishment Report\n")
    report.write("Misclassified Samples:\n\n")

    report.write(
        "Feature No.                            Feature Values                            Actual Class           Predicted Class\n")

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

        xVector = np.array(xVector)

        dotProdValue = np.dot(weightVector, xVector)

        # checking the value of w*X. if w*x > 0 then class 1 otherwise class 2
        if dotProdValue > 0:
            predictedClass = 1
        else:
            predictedClass = 2

        className = int(className)
        predictedClass = int(predictedClass)

        if (className == predictedClass):
            correctClassifyCounter = correctClassifyCounter + 1
        else:
            report.write(str(i + 1) + "              " + str(lineData[:no_of_features]) + "             "
                         + str(className) + "                     " + str(predictedClass) + "\n")


    report.write("\n")
    print("Reward and Punishement CorrectClassify: ", correctClassifyCounter)

    accuracy = (correctClassifyCounter / no_of_samples) * 100

    print("Reward and Punishement  Accuracy: ", accuracy, "%\n")
    report.write("Accuracy: " + str(accuracy) + "%\n")


def PocketAlgo():
    f = open("./datasets/trainLinearlyNonSeparable.txt", "r")
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
    print("Pocket Algo Trainning")

    # defining the value of rho
    rho = 0.1
    # creating the W(t) matrix
    weightVector = []

    for i in range(no_of_features + 1):
        weightVector.append(0.0)

    weightVector = np.array(weightVector)

    #initializing Ws as WeightVector and hs to 0
    Ws = weightVector
    hs=0

    for i in range(maxIteration):

        # defining the misclassified feature vector set
        misClassified = []

        for j in range(no_of_samples):

            lineData = lines[j].split()

            className = lineData[no_of_features]

            className = int(className)

            # creating feature vector
            featureVector = lineData[:no_of_features]

            # append 1 in the end of the feature vector
            featureVector.append(1.0)

            # converting the values of feature vector to float
            for k in range(no_of_features + 1):
                featureVector[k] = float(featureVector[k])

            featureVector = np.array(featureVector)

            dotValue = np.dot(weightVector, featureVector)

            if (dotValue >= 0):
                experimentalClass = 1
            else:
                experimentalClass = 2

            # checking if misclassified and multiplying delX with feature vector
            if (experimentalClass == 1 and className == 2):
                delX = 1
                featureVector = featureVector * delX
                misClassified.append(featureVector)

            elif (experimentalClass == 2 and className == 1):
                delX = -1
                featureVector = featureVector * delX
                misClassified.append(featureVector)


        #updating the valus of hs
        correctlyClassified = no_of_samples - len(misClassified)
        if( hs < correctlyClassified):
            hs = correctlyClassified
            ws = weightVector

        # Checking all got classified or not.When no misclassified then break the iteration.
        if (len(misClassified) == 0):
            break

        # updating the Weight vector
        sumVector = []
        for k in range(no_of_features + 1):
            sumVector.append(0.0)

        sumVector = np.array(sumVector)

        for k in range(len(misClassified)):
            sumVector += misClassified[k] * rho

        weightVector = weightVector - sumVector

    # testing of data set

    print("Pocket Algo Testing")

    f2 = open("./datasets/testLinearlyNonSeparable.txt", "r")

    report = open("Pocket Algo Report.txt","w")
    report.write("Pocket Algo Report\n")
    report.write("Misclassified Samples:\n\n")

    report.write(
        "Feature No.                            Feature Values                            Actual Class           Predicted Class\n")

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

        xVector = np.array(xVector)

        dotProdValue = np.dot(ws, xVector)

        # checking the value of w*X. if w*x > 0 then class 1 otherwise class 2
        if dotProdValue > 0:
            predictedClass = 1
        else:
            predictedClass = 2

        className = int(className)
        predictedClass = int(predictedClass)

        if (className == predictedClass):
            correctClassifyCounter = correctClassifyCounter + 1
        else:
            report.write(str(i + 1) + "              " + str(lineData[:no_of_features]) + "             "
                         + str(className) + "                     " + str(predictedClass) + "\n")


    report.write("\n")
    print("Pocket Algo CorrectClassify: ", correctClassifyCounter)

    accuracy = (correctClassifyCounter / no_of_samples) * 100

    print("Pocket Algo Accuracy: ", accuracy, "%\n")
    report.write("Accuracy: " + str(accuracy) + "%\n")


#calling the algorithms
BasicParceptron()
RewardAndPunishment()
PocketAlgo()