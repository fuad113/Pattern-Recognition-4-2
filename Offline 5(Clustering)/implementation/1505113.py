import numpy as np
import math
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(1000000)

#---------------------------------------------------------
#reading the dataset

#f = open("data/moons.txt", "r")
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


def DFSserching(point , clusterID):
    cluster[point] = clusterID
    visited[point] = True

    for i in range(datasetLength):
        dis = euclideanDistance(dataset[point],dataset[i])
        if( visited[i]==False and dis<=eps ):
            DFSserching(i , clusterID)


def DBSCAN(eps):
    print("Running DBSCAN")

    #finding the core points
    #array for saving the corePoints
    corePoints = []

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
            corePoints.append(i)

    #print("corepoints size=" , len(corePoints))

    #now running DFS to find clusters

    clusterID = 0

    for point in corePoints:
        if(visited[point] == True):
            continue

        clusterID = clusterID+1
        DFSserching(point , clusterID)

    print("Number of Clusters:",clusterID)

    #Now plotting the clusters
    #define colors for various clusters
    colors = ["#922B21" , "#9B59B6" , "#1F618D" , "#48C9B0" , "#0B5345" , "#229954" , "#F1C40F" , "#9C640C" , \
              "#DC7633" , "#979A9A" , "#424949" , "#273746" , "#D5D8DC" , "#A9CCE3" , "#707B7C" , "#E74C3C"]

    for i in range(datasetLength):
        if(cluster[i]!=0):
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

#moons.txt eps = 0.06
#bisecting.txt eps = 0.4
#blobs.txt eps = 0.75
#---------------------------------------------------------