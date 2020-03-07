import numpy as np


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



