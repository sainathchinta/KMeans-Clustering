import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os

data = pd.read_csv("/Users/sai/Documents/k-means/faithful.csv")
print(data.columns.tolist())

X = data[["eruptions","waiting"]].values
print(X.shape)


K = 2


# randomly select 2 points calculate eculdian distance of of these
# suppose mean (x1,y1)
# value = some z need to z is close to x1 or y1 , if closed to x1 assign and have a new mean

np.random.seed(0)
centroids = X[np.random.choice(len(X), K, replace=False)]
print("intial centroid:\n" , centroids)


def distance(a,b):
    return np.sqrt(np.sum((a - b) ** 2))


def assign_clusters(X, centroids):
    labels = []
    for x in X:
        dists = []
        # calculate distance from this point to each centroid
        dists = [distance(x, c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)

def update_centroids(X, labels, K):
    new_centroids = []
    for k in range(K):
        cluster_points = X[labels == k]       # all points in cluster k
        new_centroids.append(cluster_points.mean(axis=0))  # mean of points
    return np.array(new_centroids)




for i in range(10):
    labels = assign_clusters(X,centroids)
    centroid = update_centroids(X,labels,K)


plt.figure()
plt.scatter(
    X[:,0], X[:,1],
    c=labels,                # color by cluster
    cmap="viridis",
    s=80,                     # size of dots
    edgecolors="black"        # border of dots
)

# mark centroids
plt.scatter(
    centroids[:,0], centroids[:,1],
    c="red",
    s=200,
    marker="X",
    label="Centroids"
)

plt.xlabel("Eruptions")
plt.ylabel("Waiting")
plt.title("K-Means Clustering (K=2)")
plt.legend()
plt.savefig("kmeans_result.png")   # save plot
plt.show()

    