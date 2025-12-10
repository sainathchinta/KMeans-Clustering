**K-Means Clustering on Old Faithful Dataset**

**Algorithm**
1. Randomly select K points as initial centroids. These are the starting "means" of each cluster

2. For each data point:
Calculate distance from the point to each centroid
Assign the point to the cluster whose centroid is closest

3. After all points are assigned:
Recalculate each centroid as the mean of all points in that cluster

4. Repeat steps 2-3 until centroids do not move much(or a fixed number of iterations , in this case I am doing for 10 iterations)
   
6. Done: each point belongs to a cluster, centroids are final


This project implements the K-Means clustering algorithm in Python and applies it to the Old Faithful geyser dataset (faithful.csv). The project includes data exploration, visualization, and clustering using Python packages such as NumPy, Pandas, and Matplotlib.

Project Overview

**Dataset:**

faithful.csv contains eruption durations and waiting times for the Old Faithful geyser.

The dataset is used to identify natural clusters in the data.

**Steps:**

Load the dataset using Pandas.

Explore the data: check shape, basic statistics, and column names.

Visualize the data in a 2D scatter plot using Matplotlib.

Save the plot as a PNG file for reference.

**K-Means Clustering:**

Apply the K-Means algorithm to the dataset.

Initialize cluster centroids.

Assign data points to the nearest centroid based on Euclidean distance.

Update centroids iteratively until convergence.

Visualize the clusters along with centroid markers.
