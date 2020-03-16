import numpy as np
import math


# creating object for each classtype

class classobject:
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

    def calculate_mean(self):
        for i in range(self.num_of_features):
            temp = []
            for j in range(len(self.features_array)):
                temp.append(self.features_array[j][i])

            x = np.mean(np.array(temp).astype(float))
            self.mean_array.append(x)


    def calculate_sd(self):
        for i in range(self.num_of_features):
            temp = []
            for j in range(len(self.features_array)):
                temp.append(self.features_array[j][i])

            x = np.std(np.array(temp).astype(float))
            self.sd_array.append(x)

    def noofelementsinclass(self):
        temp=len(self.features_array)
        return temp


    def calculatecovarmat(self):
        self.covarmat=[[float(0) for j in range(no_of_features)] for i in range(no_of_features)]

        for i in range(no_of_features):
            for j in range(no_of_features):
                sum=0.0
                for k in range(len(self.features_array)):
                    temp1= (self.features_array[k][i]-self.mean_array[i])
                    temp2= (self.features_array[k][j]-self.mean_array[j])
                    sum+= temp1*temp2

                self.covarmat[i][j]=sum/ len(self.features_array)


# dictionary for maintaining classes

class_dictionary = {}

# reading the file and getting the no. of features , classes and samples given

f = open("./during evaluation/Train.txt", "r")
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

    if class_name in class_dictionary:
        class_dictionary[class_name].features_array.append(lines_data[:no_of_features])

    else:
        class_dictionary[class_name] = classobject(class_name,no_of_features)
        class_dictionary[class_name].features_array.append(lines_data[:no_of_features])



# training of the dataset
for i in class_dictionary:
    ob= class_dictionary[i]
    ob.calculate_mean()
    ob.calculate_sd()
    ob.convertofloat()
    #ob.printobject()
    ob.calculatecovarmat()


# matrix multiplication function
def matrixmul(mat1=[],mat2=[]):
    m1= np.array(mat1)
    m2= np.array(mat2)
    result=np.dot(m1,m2)
    return result

# transpose of a matrix
def tranposemat(mat=[]):
    m=np.array(mat)
    result= m.transpose()
    return result

# deteminant of a matrix
def determinantmat(mat=[]):
    m=np.array(mat)
    result= np.linalg.det(m)
    return result

# inverse of a matrix
def inversemat(mat=[]):
    m=np.array(mat)
    result= np.linalg.inv(m)
    return result

# testing

f2 = open("./during evaluation/Test.txt", "r")
test = f2.readlines()
f2.close()

correctness_counting=0


for test_line in range(no_of_samples):
    line=test[test_line].split()

    given_class=line[no_of_features]

    feat_arr=[]

    for j in range(no_of_features):
        feat_arr.append(float(line[j]))

    check=0.0
    p_class=''

    for key in class_dictionary:
        classob=class_dictionary[key]

        # calculation of the nom
        covar= classob.covarmat

        covar_det= determinantmat(covar)

        denom= pow(2 * math.pi, no_of_features/2) * pow(covar_det, 0.5)

        # calculation of the nom
        temp_mat=[]

        for i in range(no_of_features):
            temp_mat.append(feat_arr[i]-classob.mean_array[i])


        temp_mat_transpose= tranposemat(temp_mat)
        covar_inverse= inversemat(covar)

        matmulres= matrixmul(matrixmul(temp_mat_transpose,covar_inverse), temp_mat)

        nom= math.exp(-0.5* matmulres)

        p=nom/denom

        if (p >= check):
            check = p
            p_class = classob.class_name



    if(given_class==p_class):
        correctness_counting += 1


accuracy= (correctness_counting/no_of_samples)*100

print(accuracy)
