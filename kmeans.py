import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.cluster import KMeans
import random

x,colour = make_blobs(centers = 3, random_state = 42)
# sns.scatterplot(x[:,0],x[:,1])
# plt.show()
#
# model = KMeans(n_clusters = 3)
# model.fit(x)
# KMeans(n_clusters=3)
# y_gen = model.labels_
# print(y_gen)

# sns.scatterplot(x[:,0],x[:,1],y_gen)
# for center in model.cluster_centers_:
#     plt.scatter(center[0],center[1],s=60)
# plt.show()

class Clusters:
    def __init__(self, center):
        self.center = center
        self.points = []

    def distance(self,point):
        return np.sqrt(np.sum((point-self.center)**2))

class CustomKMeans:

    def __init__(self, n_clusters = 3, max_iters = 300):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self,x):
        clusters = []
        for i in range(self.n_clusters):
            cluster = Clusters(random.choice(x))
            clusters.append(cluster)

        for i in range(self.max_iters):

            labels = []
            for point in x:
                distances = []
                for cluster in clusters:
                    distances.append(cluster.distance(point))
                closest_idx = np.argmin(distances)
                closest_cluster = clusters[closest_idx]
                closest_cluster.points.append(point)
                labels.append(closest_idx)

            for cluster in clusters:
                cluster.center = np.mean(cluster.points, axis = 0)

        self.labels = labels
        self.cluster_centers_ = [cluster.center for cluster in clusters]

model = CustomKMeans(n_clusters=2)
model.fit(x)
sns.scatterplot(x[:,0],x[:,1],model.labels)
for center in model.cluster_centers_:
    plt.scatter(center[0],center[1],s=60)
plt.show()


