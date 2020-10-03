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


    def printmeanndsd(self):
        print(self.mean_array)
        print(self.sd_array)



# dictionary for maintaining classes

class_dictionary = {}

# reading the file and getting the no. of features , classes and samples given

f = open("./during coding/Train.txt", "r")
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


#testing

f2 = open("./during coding/Test.txt", "r")
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

        p_ci=len(classob.features_array)/no_of_samples

        for i in range(no_of_features):

            denom = math.sqrt(2 * math.pi * classob.sd_array[i] * classob.sd_array[i])

            temp1 = (feat_arr[i]-classob.mean_array[i]) * (feat_arr[i]-classob.mean_array[i])
            temp2 = 2 * classob.sd_array[i] * classob.sd_array[i]

            nom= math.exp(-(temp1/temp2))

            p_fj_ci = nom/denom

            p_ci *= p_fj_ci


        if (p_ci >= check):
            check=p_ci
            p_class=classob.class_name

    if(given_class==p_class):
        correctness_counting += 1


accuracy= (correctness_counting/no_of_samples)*100


print(accuracy)


















