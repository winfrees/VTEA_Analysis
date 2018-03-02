from copy import deepcopy
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
from Lib import statistics
from scipy.stats import gaussian_kde

import time
import warnings

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import os
import shutil

currdir = os.getcwd()
print(currdir)

if not os.path.exists('results1'):
    os.mkdir('results1')

os.chdir('results1')

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

tables = ['K1', 'K2', 'K3', 'K4', 'K5', 'W1', 'W2', 'W3', 'W4', 'W5']
for table in tables:
    if not os.path.exists(table):
        os.mkdir(table)
        
    os.chdir(table)
    # Importing the dataset
    data = pd.read_csv(currdir + '\\Datasets\\Flow data\\' + table + '.csv')
    print(data.shape)   # size of data
    data.head()

    # Getting the values
    cols = ['cd11c FITC', 'CD45 PE', 'F4/80 PeCy7', 'cd11b PB', 'MHCII APC', 'Ly6G APC cy7', 'PI']
    X = np.zeros(data.shape)
    for i in range(7):
        X[:, i] = data[cols[i]].values

    #print(np.mean(X, axis=0))   # mean of data
    #sigma = np.cov(np.transpose(X))
    #print('\ncovariance matrix:\n')
    #print(sigma)
    #w, v = LA.eig(sigma)
    #print('\neigenvalues:\n')
    #print(w)
    #print('\neigenvectors:\n')
    #print(v)
    #print()

    # restrict data
    X = np.array([X[j] for j in range(len(X)) if X[j, 6] < 10000 and X[j, 1] > 10000])
    # change negative values to ones so they can be logged
    X = np.maximum(X, np.ones(X.shape))

    X = np.log10(X)    # take the log of the data

    fig1, ax1 = plt.subplots(7,7, sharex = 'all', sharey = 'all')
    for m in range(7):  # row of plot grid, corresponds to y-value
        for n in range(7):  # column of plot grid, corresponds to x-value
            if m != n:
                x = X[:, n]
                y = X[:, m]
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
                ax1[m,n].scatter(x, y, c=z, s=10, edgecolor='')
                
            if m == 6:  # put x-label axis below bottom row
                ax1[m,n].set(xlabel=cols[n])

            if n == 0:  # put y-label axis next to first column
                ax1[m,n].set(ylabel=cols[m])

    plt.suptitle(table)
    plt.savefig('density.png')    # save plot
    plt.clf()
    plt.close()

                  
    # ============
    # Set up cluster parameters
    # ============

#    for k in range(3,10):
#        if not os.path.exists('k=' + str(k)):
#            os.mkdir('k=' + str(k))
#            
#        os.chdir('k=' + str(k))
#        newdir = os.getcwd()
#        params = {'quantile': .3,
#                  'eps': .3,   # higher values of eps lead to more DBSCAN clusters
#                  'damping': .9,
#                  'preference': -200,
#                  'n_neighbors': 10,
#                  'n_clusters': k}  # number of clusters (for clustering methods that require this parameter)
#        # normalize dataset for easier parameter selection
#        #X = StandardScaler().fit_transform(X)
#        
#        # estimate bandwidth for mean shift
#        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
#
#        # connectivity matrix for structured Ward
#        connectivity = kneighbors_graph(
#                X, n_neighbors=params['n_neighbors'], include_self=False)
#        # make connectivity symmetric
#        connectivity = 0.5 * (connectivity + connectivity.T)
#
#        # ============
#        # Create cluster objects
#        # ============
#        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#        ward = cluster.AgglomerativeClustering(
#                n_clusters=params['n_clusters'], linkage='ward',
#                connectivity=connectivity)
#        spectral = cluster.SpectralClustering(
#                n_clusters=params['n_clusters'], eigen_solver='arpack',
#                affinity="nearest_neighbors")
#        dbscan = cluster.DBSCAN(eps=params['eps'])
#        affinity_propagation = cluster.AffinityPropagation(
#                damping=params['damping'], preference=params['preference'])
#        average_linkage = cluster.AgglomerativeClustering(
#                linkage="average", affinity="cityblock",
#                n_clusters=params['n_clusters'], connectivity=connectivity)
#        birch = cluster.Birch(n_clusters=params['n_clusters'])
#        gmm = mixture.GaussianMixture(
#                n_components=params['n_clusters'], covariance_type='full')
#
#        # uncomment clustering algorithms to include them
#        clustering_algorithms = (
#                ('MiniBatchKMeans', two_means),
#                #('AffinityPropagation', affinity_propagation),
#                #('MeanShift', ms),
#                #('SpectralClustering', spectral),
#                #('Ward', ward),
#                #('AgglomerativeClustering', average_linkage),
#                #('DBSCAN', dbscan),
#                #('Birch', birch),
#                ('GaussianMixture', gmm)
#                )
#
#        for name, algorithm in clustering_algorithms:
#            if not os.path.exists(name):
#                os.mkdir(name)
#                
#            os.chdir(name)
#            #directory = os.getcwd()
#            centroids = np.zeros((100*k, 7))
##            for trial in range(100):
###                if not os.path.exists('trial' + str(trial)):
###                    os.mkdir('trial' + str(trial))
###                    
###                os.chdir('trial' + str(trial))
##                t0 = time.time()
##                # catch warnings related to kneighbors_graph
##                with warnings.catch_warnings():
##                    warnings.filterwarnings(
##                            "ignore",
##                            message="the number of connected components of the " +
##                            "connectivity matrix is [0-9]{1,2}" +
##                            " > 1. Completing it to avoid stopping the tree early.",
##                            category=UserWarning)
##                    warnings.filterwarnings(
##                            "ignore",
##                            message="Graph is not fully connected, spectral embedding" +
##                            " may not work as expected.",
##                            category=UserWarning)
##                    algorithm.fit(X)
##
##                t1 = time.time()
##                # y_pred is cluster number
##                if hasattr(algorithm, 'labels_'):
##                    y_pred = algorithm.labels_.astype(np.int)
##
##                else:
##                    y_pred = algorithm.predict(X)
##
##                lowerbound = int(min(y_pred))    # lowest cluster number
##                upperbound = int(max(y_pred) + 1)   # highest cluster number
##                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
##                                                     '#f781bf', '#a65628', '#984ea3',
##                                                     '#999999', '#e41a1c', '#dede00']),
##                                                     9*(upperbound - lowerbound))))
##                #print(name)
##                #print('\nnumber of clusters:')
##                #print(upperbound - lowerbound)
##                #print()
##                means = np.zeros((k, 7))
##                #f = open('clusters.txt', 'w+')
##                for i in range(k):
##                    # get cluster
##                    #fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
##                    points = np.array([X[j] for j in range(len(X)) if int(y_pred[j]) == i])
##                    size = len(points)  # size of cluster
##                    if size > 0:
##                        # save cluster info
##                        #f.write('cluster ')
##                        #f.write(str(i) + '\r\n')
##                        #f.write('size:')
##                        #f.write(str(size) + '\r\n')
##                        #f.write(str(colors[i]) + '\r\n')    # color of cluster
##                        #means[i] = np.mean(points, axis=0)
##                        centroids[k*trial+i] = np.mean(points, axis=0)
##                        #f.write(str(means[i]) + '\r\n\r\n')  # mean of cluster
##                        #print(np.std(points, axis=0))
##                        #print(np.cov(np.transpose(points)))
##                        #print()
#                    
#                        # plot each cluster individually
##                        for m in range(7):
##                            for n in range(7):
##                                ax[m,n].scatter(points[:, n], points[:, m], s=10, color=colors[i])
##                                if m == 6:
##                                    ax[m,n].set(xlabel=cols[n])
##                                
##                                if n == 0:
##                                    ax[m,n].set(ylabel=cols[m])
##                                
##                        plt.suptitle(name + ', k=' + str(k) + ', cluster ' + str(i))
##                        plt.savefig('cluster' + str(i) + '.png')    # save plot
##                        plt.clf()
##                        plt.close()
#
#                #f.close()
#                # plot data
##                fig1, ax1 = plt.subplots(7,7, sharex = 'all', sharey = 'all')
##                for m in range(7):  # row of plot grid, corresponds to y-value
##                    for n in range(7):  # column of plot grid, corresponds to x-value
##                        ax1[m,n].scatter(X[:, n], X[:, m], s=10, color=colors[y_pred])
##                        if m == 6:  # put x-label axis below bottom row
##                            ax1[m,n].set(xlabel=cols[n])
##
##                        if n == 0:  # put y-label axis next to first column
##                            ax1[m,n].set(ylabel=cols[m])
##
##                plt.suptitle(name + ' data, k=' + str(k))
##                plt.savefig('data.png')    # save plot
##                plt.clf()
##                plt.close()
#            
#                # plot cluster means
##                fig2, ax2 = plt.subplots(7,7, sharex = 'all', sharey = 'all')
##                for m in range(7):
##                    for n in range(7):
##                        ax2[m,n].scatter(means[:, n], means[:, m], s=10, color=colors)
##                        if m == 6:
##                            ax2[m,n].set(xlabel=cols[n])
##                    
##                        if n == 0:
##                            ax2[m,n].set(ylabel=cols[m])
##                    
##                plt.suptitle(name + ' centroids, k=' + str(k))
##                plt.savefig('centroids.png')    # save plot
##                plt.clf()
##                plt.close()
##                os.chdir(directory)
#                
#            #get centroids from text file
##            cfile = open('centroids.txt', 'r')
##            i = 0
##            line = cfile.readline()
##            while line:
##                line = line.replace(';', '')
##                numbers = line.split(' ')
##                for j in range(7):
##                    centroids[i,j] = float(numbers[j])
##                    
##                cfile.readline()
##                line = cfile.readline()
##                i = i + 1
##                
##                
###            for i in range(100*k):
###                for j in range(7):
###                    cfile.write(str(centroids[i,j]))
###                    cfile.write(' ')
###                    
###                cfile.write(';\r\n')
###                
##            cfile.close()
##            # plot all cluster means together
##            fig3, ax3 = plt.subplots(7,7, sharex = 'all', sharey = 'all')
##            for m in range(7):
##                for n in range(7):
##                    if m != n:
##                        x = centroids[:, n]
##                        y = centroids[:, m]
##                        xy = np.vstack([x, y])
##                        z = gaussian_kde(xy)(xy)
##                        idx = z.argsort()
##                        x, y, z = x[idx], y[idx], z[idx]
##                        ax3[m,n].scatter(x, y, c=z, s=10, edgecolor='')
##                    
##                    if m == 6:
##                        ax3[m,n].set(xlabel=cols[n])
##                        
##                    if n == 0:
##                        ax3[m,n].set(ylabel=cols[m])
##                        
##            plt.suptitle(name + ', all centroids, k=' + str(k))
##            plt.savefig('densitycentroids.png')
##            plt.clf()
##            plt.close()
##            os.chdir(newdir)
#
#        os.chdir(currdir + '\\results1\\' + table)

    os.chdir(currdir + '\\results1')
    
os.chdir(currdir)