# DBSCAN-machine-learning
Density-Based Spatial Clustering of Applications with Noise

#### Mean shift is a non-parametric function-space analysis technique for locating the maxima of a density function, a so-called mode-seeking algorithm. Application domains include cluster analysis in computer vision and image processing.

#### You need to search in the internet and see the difference between this algorithm and the other aogoritms

`` `python
# import requirements
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import numpy as np
# load data
iris = load_iris ()
# set model
dbscan = DBSCAN (eps = 0.5, min_samples = 3)
dbscan.fit (iris.data)
labels = dbscan.labels_

plt.scatter (iris.data [:, 0], iris.data [:, 1], c = labels)
plt.show ()
`` `

you can change eps and min_samples and see what happens


## Abstract Algorithm :book:
The DBSCAN algorithm can be abstracted into the following steps:

#### 1.Find the points in the ε (eps) neighborhood of each point, and identify the core points with more than minPts neighbors.
#### 2.Find the connected components of the core points on the neighbor graph, ignoring all non-core points.
#### 3. Assign each non-core point to a nearby cluster if the cluster is an ε (eps) neighbor, otherwise, assign it to noise.
A naive implementation of this requires storing the neighborhoods in step 1, thus requiring substantial memory. The original DBSCAN algorithm does not require this by performing these steps for one point at a time.

I hope this article will be useful to you.
