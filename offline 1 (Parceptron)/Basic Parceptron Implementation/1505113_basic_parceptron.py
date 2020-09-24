import numpy as np
import math


# creating object for each classtype
class ClassObject:
    def __init__(self, class_name,num_of_features):
        self.class_name = class_name
        self.num_of_features=num_of_features
        self.features_array = []
        self.mean_array=[]
        self.sd_array=[]
        self.covarmat=[]

    def convertofloat(self):
        for i in range(len(self.features_array)):
            for j in range(no_of_features):
                self.features_array[i][j]=float(self.features_array[i][j])


    def printobject(self):
        print(self.features_array)

    def noofelementsinclass(self):
        temp=len(self.features_array)
        return temp



# dictionary for maintaining classes
classDictionary = {}
# reading the file and getting the no. of features , classes and samples given

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

for i in range(no_of_samples):
    lines_data = lines[i].split()
    class_name = lines_data[no_of_features]

    if class_name in classDictionary:
        classDictionary[class_name].features_array.append(lines_data[:no_of_features])

    else:
        classDictionary[class_name] = ClassObject(class_name, no_of_features)
        classDictionary[class_name].features_array.append(lines_data[:no_of_features])

#converting string to float in dictionary
for i in classDictionary:
    ob= classDictionary[i]
    ob.convertofloat()


# matrix multiplication function
def matrixmuldot(mat1=[],mat2=[]):
    m1= np.array(mat1)
    m2= np.array(mat2)
    result=np.dot(m1,m2)
    return result

# transpose of a matrix
def tranposemat(mat=[]):
    m=np.array(mat)
    result= m.transpose()
    return result

# training of the dataset
#defining the value of rho
rho=1

#creating the W(t) matrix
weightMatrix=[]

for i in range(no_of_features):
    weightMatrix.append(0.0)










