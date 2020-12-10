import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import random

sys.setrecursionlimit(1000000)

maxIteration =10000

#---------------------------------------------------------
#reading the dataset

#f= open("data/moons.txt", "r")
#f = open("data/blobs.txt", "r")
f = open("data/bisecting.txt", "r")

lines = f.readlines()
f.close()

dataset = []

for l in lines:
    line = l.split()
    feat1 = float(line[0])
    feat2 = float(line[1])
    tempArray =[]
    tempArray.append(feat1)
    tempArray.append(feat2)
    dataset.append(tempArray)

print("Dataset Length:",len(dataset))
#---------------------------------------------------------

#---------------------------------------------------------
#function for calculating euclidean distance between 2 points

def euclideanDistance(point1 , point2):
    temp1 = pow((point1[0]-point2[0]), 2)
    temp2 = pow((point1[1]-point2[1]), 2)
    temp = temp1 + temp2
    distance = math.sqrt( temp )

    return distance
#---------------------------------------------------------

#---------------------------------------------------------
#work for calculating eps
datasetLength = len(dataset)

def calculateEPS(k):
    #take an array to hold the kth neighbour distance for every points in the dataset
    kDistance=[]

    for i in range(datasetLength):
        tempArray = []
        for j in range(datasetLength):
            dis = euclideanDistance(dataset[i],dataset[j])

            if(dis == 0):
                #denotes the distance between same points
                continue
            else:
                tempArray.append(dis)

        #sorting the tempArray
        tempArray.sort()
        kthDistance = tempArray[k-1]
        kDistance.append(kthDistance)

    kDistance.sort()
    #now plot the graph
    plt.figure(1)
    plt.title("Finding EPS")
    plt.plot(kDistance, color="#000000")
    plt.grid()
    plt.xlabel("Points Sorted According the Distance of "+str(k)+"th Nearest Neighbour")
    plt.ylabel(str(k)+"th Nearest Neighbour Distance")
    plt.show()

#---------------------------------------------------------

#---------------------------------------------------------
#arrays for DFS
visited =[]
for i in range(datasetLength):
    visited.append(False)

cluster = []
for i in range(datasetLength):
    cluster.append(0)

noOfClusters = 0

def DFSserching( key ,point , clusterID):
    cluster[key] = clusterID
    visited[key] = True

    for i in range(datasetLength):
        dis = euclideanDistance(point,dataset[i])
        if( visited[i]==False and dis<=eps ):
            newPoint = dataset[i]
            DFSserching(i, newPoint , clusterID)


def DBSCAN(eps):
    print("Running DBSCAN")

    #finding the core points
    #array for saving the corePoints
    corePoints = {}

    for i in range(datasetLength):
        neighbourPoints = 0

        for j in range(datasetLength):
            dis = euclideanDistance(dataset[i],dataset[j])
            if(dis == 0):
                #denotes the distance between same points
                continue
            else:
                if(dis <= eps ):
                    neighbourPoints += 1

        if(neighbourPoints >= MINPTS ):
            #saving only the index of the core point
            key = i
            corePoints[key] = dataset[i]

    #print("corepoints size=" , len(corePoints))

    #now running DFS to find clusters
    clusterID = 0

    for key in corePoints:
        if(visited[key] == True):
            continue

        clusterID = clusterID+1
        point= corePoints[key]

        DFSserching(key , point ,clusterID)

    global noOfClusters

    noOfClusters= clusterID
    print("Number of Clusters:",noOfClusters)

    #Now plotting the clusters
    #define colors for various clusters
    colors = ["#922B21" , "#9B59B6" , "#1F618D" , "#48C9B0" , "#0B5345" , "#229954" , "#F1C40F" , "#9C640C" , \
              "#DC7633" , "#979A9A" , "#424949" , "#273746" , "#D5D8DC" , "#A9CCE3" , "#707B7C" , "#E74C3C"]

    plt.figure(2)
    plt.title("DBSCAN")

    for i in range(datasetLength):
        if(cluster[i] > 0):
            plotColor = colors[cluster[i]]
            feat1 = dataset[i][0]
            feat2 = dataset[i][1]
            plt.scatter(feat1, feat2, color=plotColor)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

#---------------------------------------------------------

#---------------------------------------------------------
def KMeans(noOfClusters):
    print("Running K-means")

    #initialization. Randomly choose k(no of clusters) points
    randomPointIndex = []
    # for i in range(noOfClusters):
    #     randomPointIndex.append(random.randint(0,datasetLength-1))

    #randomly choosing points is not working properly. So, we are taking points after a fixed interval
    inter = datasetLength // noOfClusters
    for i in range(noOfClusters):
        index = i*inter
        randomPointIndex.append(index)

    #print(randomPointIndex)
    centroids =[]

    for i in range(noOfClusters):
        centroids.append(dataset[randomPointIndex[i]])

    #print(centroids)

    #array for denoting which data belongs to which cluster
    cluster =[]
    for i in range(datasetLength):
        cluster.append(0)

    #distance from the centroid to all other data
    distance = []
    for i in range(datasetLength):
        distance.append(np.inf)


    for iteration in range(maxIteration):
        #for all data find which is the closest centroid
        #clearing the distance array
        for i in range(datasetLength):
            distance[i]=np.inf

        for i in range(noOfClusters):
            centroid = centroids[i]

            for j in range(datasetLength):

                point = dataset[j]

                dis =euclideanDistance(centroid , point)

                if(dis < distance[j]):
                    distance[j] = dis
                    cluster[j] = i

        #work for finding out the new centroid
        clusterList = {}

        for i in range(noOfClusters):
            if( i not in clusterList):
                tempArray=[]
                clusterList[i]=tempArray

        for i in range(datasetLength):
            key = cluster[i]
            clusterList[key].append(dataset[i])


        newCentroids = []

        for key in clusterList:
            pointsArr = clusterList[key]
            pointsArr = np.array(pointsArr)
            cent = np.average(pointsArr , axis=0)
            newCentroids.append(cent)


        #checking if we should continue or not

        diffArray = []

        for i in range(noOfClusters):
            differ=np.abs(centroids[i] - newCentroids[i])
            diffArray.append(differ)

        flag= False

        for i in range(noOfClusters):
            for j in range(2):
                if(diffArray[i][j] != 0):
                    flag= True

        if(flag == False):
            print("Converges at iteration:",iteration+1)
            break

        for i in range(noOfClusters):
            centroids[i]= newCentroids[i]


    #Now plotting the clusters
    #define colors for various clusters
    colors = ["#922B21" , "#9B59B6" , "#1F618D" , "#48C9B0" , "#0B5345" , "#229954" , "#F1C40F" , "#9C640C" , \
              "#DC7633" , "#979A9A" , "#424949" , "#273746" , "#D5D8DC" , "#A9CCE3" , "#707B7C" , "#E74C3C"]


    plt.figure(3)
    plt.title("K-means")

    for i in range(datasetLength):
        plotColor = colors[cluster[i]]
        feat1= dataset[i][0]
        feat2= dataset[i][1]
        plt.scatter(feat1,feat2,color = plotColor)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

#---------------------------------------------------------

#---------------------------------------------------------
#calculate the EPS for value k=4 and then run DBSCAN
MINPTS = 4

calculateEPS(MINPTS)
input = input("What is the value of EPS:")
eps =float(input)
print("EPS: ",eps)
DBSCAN(eps)

#parameter is the number of clusters
KMeans(noOfClusters)

#---------------------------------------------------------
#moons.txt eps = 0.06
#blobs.txt eps = 1.0
#bisecting.txt eps = 0.45
