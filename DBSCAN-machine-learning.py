# import requirements
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import numpy as np
# load data
iris = load_iris()
# set model
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(iris.data)
labels = dbscan.labels_

plt.scatter(iris.data[:,0], iris.data[:,1], c=labels)
plt.show()




