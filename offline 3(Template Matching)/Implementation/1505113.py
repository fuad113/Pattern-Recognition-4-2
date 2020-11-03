import cv2
import numpy as np
import time

#read the reference image in gray Scale
referenceImage = cv2.imread("reference.jpg" , 0)
shapeOfReferenceImage = referenceImage.shape

M=shapeOfReferenceImage[0]
N= shapeOfReferenceImage[1]

#set the value of p
p=3

#-----------------------------------------------------------------------------------
#read the video
frames=[]

# Capture the frames of the input video to an array of frames
cap = cv2.VideoCapture("input.mov")
fps = cap.get(cv2.CAP_PROP_FPS)

flag = 1

while flag:
    flag,frame = cap.read()

    if(flag == 0):
        break

    grayFrame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    frames.append(grayFrame)

cap.release()
numOfFrames = len(frames)
#-----------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------
#Exhaustive search
startingTime = time.time()
outputFrames =[]

exhaustiveSearching = 0

#exhasutive working with the first frame of the video
firstFrameShape= frames[0].shape

I= firstFrameShape[0]
J= firstFrameShape[1]

prevI = -1
PrevJ = -1

#working of the 1st frame
minD = np.inf
markedIIndex = 0
markedJIndex = 0

iStart = 0
iEnd = I-M+1
jStart = 0
jEnd = J-N+1

frame = frames[0]

for i in range(iStart, iEnd):
    for j in range(jStart, jEnd):
        # calculate the D(m,n)
        # first make a sub matrix
        subMatrix = frame[i:i + M, j:j + N]
        subMatrix = subMatrix.astype(np.int64)

        # calculation of difference
        tempRef = referenceImage.astype(np.int64)
        difference = tempRef - subMatrix
        difference = np.absolute(difference)

        diffSquare = difference * difference

        d = np.sum(diffSquare)

        exhaustiveSearching += 1

        if (d < minD):
            minD = d
            markedIIndex = i
            markedJIndex = j

# update the prevI and prevJ
prevI = markedIIndex
prevJ = markedJIndex

# convert the frame to rgb
rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
# draw the rectangle around the reference image
cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
# append to the output frame
outputFrames.append(rgbFrame)



#now working with the other frames
for index in range(1, numOfFrames ,1):

    iStart = prevI - p
    iEnd = prevI + p
    jStart = prevJ - p
    jEnd = prevJ + p

    minD = np.inf
    markedIIndex = 0
    markedJIndex = 0

    frame = frames[index]

    for i in range(iStart, iEnd):
        for j in range(jStart, jEnd):
            if ( (i < 0) or (i > I - M) or (j < 0) or (j > J - N)):
                continue

            # calculate the D(m,n)
            # first make a sub matrix
            subMatrix = frame[i:i + M, j:j + N]
            subMatrix = subMatrix.astype(np.int64)

            # calculation of difference
            tempRef = referenceImage.astype(np.int64)
            difference = tempRef - subMatrix
            difference = np.absolute(difference)

            diffSquare = difference * difference

            d = np.sum(diffSquare)

            exhaustiveSearching += 1

            if (d < minD):
                minD = d
                markedIIndex = i
                markedJIndex = j


    # update the prevI and prevJ
    prevI = markedIIndex
    prevJ = markedJIndex

    # convert the frame to rgb
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # draw the rectangle around the reference image
    cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
    # append to the output frame
    outputFrames.append(rgbFrame)


endingTime = time.time()

timeDuration =endingTime -startingTime

#now make the output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("1505113_Exhaustive.mov", fourcc, fps, (J, I))

for frame in outputFrames:
    output.write(frame)

output.release()

print("Exhasutive search")
print("-----------------")
print("Value of P: " , p)
print("Time Taken: ", timeDuration ,"seconds")
print("Searching Done: " ,exhaustiveSearching)

#-----------------------------------------------------------------------------------





