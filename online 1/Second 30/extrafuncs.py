import numpy as np
import math as math

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


matrix1 = [[12,7,3],
        [4 ,5,6],
        [7 ,8,9]]
matrix2 = [[5,8,1],
        [6,7,3],
        [4,5,9]]
matrix3 = [[ 1,1 ,1]
, [ 0, 2 ,5]
 ,[ 2, 5, -1]]

res= math.log(math.exp,math.exp)




