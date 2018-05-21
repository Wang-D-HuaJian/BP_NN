from sklearn.datasets import load_iris
from numpy import *
dataset_iris = load_iris()
data_iris = dataset_iris['data']
label_iris = dataset_iris['target']
print(data_iris)
print(label_iris)
print(len(set(label_iris)))
#for key, val in dataset_iris.items():
# f = open("data_iris.txt", "w")
# m, n = shape(data_iris)
# print()
# f.write(data_iris)
# f.close()