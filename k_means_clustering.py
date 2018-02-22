from copy import deepcopy
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
from Lib import statistics

import time
import warnings

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('Datasets\Flow data\K1.csv')
print(data.shape)   # size of data
data.head()

# Getting the values
cols = ['cd11c FITC', 'CD45 PE', 'F4/80 PeCy7', 'cd11b PB', 'MHCII APC', 'Ly6G APC cy7', 'PI']
X = np.zeros(data.shape)
for i in range(7):
    X[:, i] = data[cols[i]].values

print(np.mean(X, axis=0))   # mean of data
sigma = np.cov(np.transpose(X))
print('\ncovariance matrix:\n')
print(sigma)
w, v = LA.eig(sigma)
print('\neigenvalues:\n')
print(w)
print('\neigenvectors:\n')
print(v)
print()

# restrict data jhgfjgjkhgkj
X = np.array([X[j] for j in range(len(X)) if X[j, 6] < 10000 and X[j, 1] > 10000])
# change negative values to ones so they can be logged
X = np.maximum(X, np.ones(X.shape))

X = np.log10(X)    # take the log of the data


# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

                  
# ============
# Set up cluster parameters
# ============

for k in range(3,10,3):
    params = {'quantile': .3,
              'eps': .3,   # higher values of eps lead to more DBSCAN clusters
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': k}  # number of clusters (for clustering methods that require this parameter)
    # normalize dataset for easier parameter selection
    #X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
    spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')

    # uncomment clustering algorithms to include them
    clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            #('AffinityPropagation', affinity_propagation),
            #('MeanShift', ms),
            #('SpectralClustering', spectral),
            #('Ward', ward),
            #('AgglomerativeClustering', average_linkage),
            #('DBSCAN', dbscan),
            #('Birch', birch),
            ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
            warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        # y_pred is cluster number
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        lowerbound = int(min(y_pred))    # lowest cluster number
        upperbound = int(max(y_pred) + 1)   # highest cluster number
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                             9*(upperbound - lowerbound))))
        print(name)
        print('\nnumber of clusters:')
        print(upperbound - lowerbound)
        print()
        fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
        means = np.zeros((k, 7))
        for i in range(k):
            # get cluster
            points = np.array([X[j] for j in range(len(X)) if int(y_pred[j]) == i])
            size = len(points)  # size of cluster
            if size > 0:
                print('cluster:')
                print(i)
                print('size:')
                print(size)
                print(colors[i])    # color of cluster
                means[i] = np.mean(points, axis=0)
                print(means[i])  # mean of cluster
                #print(np.std(points, axis=0))
                #print(np.cov(np.transpose(points)))
                print()

        for m in range(7):  # row of plot grid, corresponds to y-value
            for n in range(7):  # column of plot grid, corresponds to x-value
                ax[m,n].scatter(X[:, n], X[:, m], s=10, color=colors[y_pred])
                if m == 6:  # put x-label axis below bottom row
                    ax[m,n].set(xlabel=cols[n])

                if n == 0:  # put y-label axis next to first column
                    ax[m,n].set(ylabel=cols[m])

        plt.suptitle(name)
        plt.show()
        fig2, ax2 = plt.subplots(7,7, sharex = 'all', sharey = 'all')
        for m in range(7):
            for n in range(7):
                ax2[m,n].scatter(means[:, n], means[:, m], s=10, color=colors)
                if m == 6:
                    ax2[m,n].set(xlabel=cols[n])
                    
                if n == 0:
                    ax2[m,n].set(ylabel=cols[m])
                    
        plt.suptitle(name)
        plt.show()
                
                  