# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from copy import deepcopy
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
from Lib import statistics
from scipy.stats import gaussian_kde
import array

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

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
   
# Getting the values
original_cols = ['Ch1 mean',
        'Ch2_d1 mean',
        'Ch3_d1 mean',
        'Ch4_d1 mean',
        'Ch5_d1 mean',
        'Ch6_d1 mean',
        'Ch7_d1 mean'#,
#        'Ch1 #pixels',
#        'Ch1 sum',
#        'Ch1 max',
#        'Ch1 SD',
#        'Ch1 AR',
#        'Ch1 F_min',
#        'Ch1 F_max',
#        'Ch1 mean_th'#,
        ##'Ch1 mean_sq',
        ##'Ch1_d0 mean',
        ##'Ch1_d0 sum',
        ##'Ch1_d0 max',
        #'Ch1_d0 SD'#,
        ##'Ch1_d0 mean_th',
        ##'Ch1_d0 mean_sq'
        ]

trimmed_cols = [#'Unnamed: 0',
                'Ch1.mean',
                'Ch2_d1.mean',
                'Ch3_d1.mean',
                'Ch4_d1.mean',
                'Ch5_d1.mean',
                'Ch6_d1.mean',
                'Ch7_d1.mean'#,
#                'Ch1..pixels',
#                'Ch1.sum',
#                'Ch1.max',
#                'Ch1.SD',
#                'Ch1.AR',
#                'Ch1.F_min',
#                'Ch1.F_max',
#                'Ch1.mean_th'#,
                ##'Ch1.mean_sq',
                ##'Ch1_d0.mean',
                ##'Ch1_d0.sum',
                ##'Ch1_d0.max',
                #'Ch1_d0.SD'#,
                ##'Ch1_d0.mean_th',
                ##'Ch1_d0.mean_sq'
                ]

cols = [trimmed_cols, original_cols]
#print(cols)
#print()
#print(cols[0])
#print()
#print(cols[1])
#print()
#print(cols[0][0])
#print(cols[1][5])
tables = ['Datasets\\VTEA_trimmed.csv', 'Datasets\\VTEA.csv']
folders = ['VTEA_trimmed_plots', 'VTEA_plots']

for tablenum in range(2):
    numcols = len(cols[tablenum])
    # Importing the dataset
    data = pd.read_csv(tables[tablenum])
    #print(data.shape)   # size of data
    #print(data)
    #data.head()

    if not os.path.exists(folders[tablenum]):
        os.mkdir(folders[tablenum])

    os.chdir(folders[tablenum])

    X = np.zeros((len(data), numcols))
    for i in range(numcols):
        X[:, i] = data[cols[tablenum][i]].values
    #print(X.shape)
    #print(X)
#    for m in range(len(cols[tablenum])):
#        for n in range(m+1, len(cols[tablenum])):
#            if np.array_equal(X[:, m], X[:, n]):
#                print(cols[tablenum][m] + ' (' + str(m) + ') is '
#                      + cols[tablenum][n] + ' (' + str(n) + ')')           
#            else:
#                if np.array_equal(X[:, m] ** 2, X[:, n]):
#                    print(cols[tablenum][m] + ' (' + str(m) + ') squared is '
#                          + cols[tablenum][n] + ' (' + str(n) + ')')
#                else:
#                    if np.array_equal(X[:, m], X[:, n] ** 2):
#                        print(cols[tablenum][m] + ' (' + str(m) + ') is '
#                              + cols[tablenum][n] + ' (' + str(n) + ') squared')
#    
    if not os.path.exists('7D'):
        os.mkdir('7D')
    
    os.chdir('7D')

    tabledir = os.getcwd()

    #print(X)
    ctype = data['Object'].values
    #print(ctype)
    
    X1 = np.array([X[j] for j in range(len(X)) if ctype[j] == 'Normal'])
    X2 = np.array([X[j] for j in range(len(X)) if ctype[j] == 'Disease A'])

#    colors = ['b']*len(ctype)
#    for i in range(len(ctype)):
#        if ctype[i] == 'Disease A':
#            colors[i] = 'r'
        
    #print(colors)
    
    for k in range(3,10):
        if not os.path.exists('k=' + str(k)):
            os.mkdir('k=' + str(k))
            
        os.chdir('k=' + str(k))
        print('\nk=' + str(k) + '\n')
        kdir = os.getcwd()
        params = {'quantile': .3,
                  'eps': .3,   # higher values of eps lead to more DBSCAN clusters
                  'damping': .9,
                  'preference': -200,
                  'n_neighbors': 10,
                  'n_clusters': k}  # number of clusters (for clustering methods that require this parameter)
        # normalize dataset for easier parameter selection
        #X = StandardScaler().fit_transform(X)
        
        # estimate bandwidth for mean shift
        #bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
#        connectivity = kneighbors_graph(
#                X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        #connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        #ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#        ward = cluster.AgglomerativeClustering(
#                n_clusters=params['n_clusters'], linkage='ward',
#                connectivity=connectivity)
        spectral = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        affinity_propagation = cluster.AffinityPropagation(
                damping=params['damping'], preference=params['preference'])
#        average_linkage = cluster.AgglomerativeClustering(
#                linkage="average", affinity="cityblock",
#                n_clusters=params['n_clusters'], connectivity=connectivity)
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
            if not os.path.exists(name):
                os.mkdir(name)
                
            os.chdir(name)
            print(name)
            algdir = os.getcwd()
            sets = [X, X1, X2]
            setnames = ['All', 'Normal', 'Disease A']
            for index in range(3):
                if not os.path.exists(setnames[index]):
                    os.mkdir(setnames[index])
                    
                os.chdir(setnames[index])
                print(setnames[index])
                centroids = np.zeros((100*k, numcols))
                cfile = open('clusters.txt', 'w+')
                for trial in range(100):
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
                        algorithm.fit(sets[index])
    
                    t1 = time.time()
                    # y_pred is cluster number
                    if hasattr(algorithm, 'labels_'):
                        y_pred = algorithm.labels_.astype(np.int)
    
                    else:
                        y_pred = algorithm.predict(sets[index])
    
                    lowerbound = int(min(y_pred))    # lowest cluster number
                    upperbound = int(max(y_pred) + 1)   # highest cluster number
                    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                         '#f781bf', '#a65628', '#984ea3',
                                                         '#999999', '#e41a1c', '#dede00']),
                                                         9*(upperbound - lowerbound))))
                    #print(name)
                    #print('\nnumber of clusters:')
                    #print(upperbound - lowerbound)
                    #print()
                    means = np.zeros((k, numcols))
                    for i in range(k):
                        # get cluster
                        points = np.array([sets[index][j] for j in range(len(sets[index])) if int(y_pred[j]) == i])
                        size = len(points)  # size of cluster
                        if size > 0:
                            # save cluster info
                            #means[i] = np.mean(points, axis=0)
                            centroids[k*trial+i] = np.mean(points, axis=0)
                            for j in range(numcols):
                                cfile.write(str(centroids[k*trial+i, j]))
                                cfile.write(' ')
                                
                            cfile.write(';\r\n')
    
                cfile.close()
                # plot all cluster means together
                fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
                for m in range(numcols):
                    for n in range(numcols):
                        if m != n:
                            x = centroids[:, m]
                            y = centroids[:, n]
                            xy = np.vstack([x, y])
                            z = gaussian_kde(xy)(xy)
                            idx = z.argsort()
                            x, y, z = x[idx], y[idx], z[idx]
                            ax[n,m].scatter(x, y, c=z, s=10, edgecolor='')
                            
                        if n == 6:
                            ax[n,m].set(xlabel=cols[tablenum][m])
                            
                        if m == 0:
                            ax[n,m].set(ylabel=cols[tablenum][n])
                            
                plt.suptitle(name + ', all centroids, k=' + str(k))
                plt.savefig('densitycentroids.png')
                plt.clf()
                plt.close()
                
                os.chdir(algdir)
                
            os.chdir(kdir)

        os.chdir(tabledir)
    
    os.chdir(currdir)    
    
    
    
    
    
    
    
    
#
#    #plot all cells
#    if not os.path.exists('data'):
#        os.mkdir('data')
#    
#    os.chdir('data')
#
#    for m in range(len(cols[tablenum])):  # row of plot grid, corresponds to y-value
#        for n in range(m+1, len(cols[tablenum])):  # column of plot grid, corresponds to x-value
#            plt.scatter(X[:, m], X[:, n], s=50, color = colors)
#            plt.xlabel(cols[tablenum][m])
#            plt.ylabel(cols[tablenum][n])
#            plt.savefig('axes' + str(m) + 'and' + str(n) + '.png')    # save plot
#            plt.clf()
#            plt.close()
#        
#    os.chdir(newdir)
#
#
#    #plot normal cells
#    if not os.path.exists('Normal'):
#        os.mkdir('Normal')
#    
#    os.chdir('Normal')
#
#    for m in range(len(cols[tablenum])):  # row of plot grid, corresponds to y-value
#        for n in range(m+1, len(cols[tablenum])):  # column of plot grid, corresponds to x-value
#            plt.scatter(X1[:, m], X1[:, n], s=50, color='b')
#            plt.xlabel(cols[tablenum][m])
#            plt.ylabel(cols[tablenum][n])
#            plt.savefig('axes' + str(m) + 'and' + str(n) + '.png')    # save plot
#            plt.clf()
#            plt.close()
#
#    os.chdir(newdir)
#
#    # plot diseased cells
#    if not os.path.exists('Disease A'):
#        os.mkdir('Disease A')
#    
#    os.chdir('Disease A')
#
#    for m in range(len(cols[tablenum])):  # row of plot grid, corresponds to y-value
#        for n in range(m+1, len(cols[tablenum])):  # column of plot grid, corresponds to x-value
#            plt.scatter(X2[:, m], X2[:, n], s=50, color='r')
#            plt.xlabel(cols[tablenum][m])
#            plt.ylabel(cols[tablenum][n])
#            plt.savefig('axes' + str(m) + 'and' + str(n) + '.png')    # save plot
#            plt.clf()
#            plt.close()
#
#    os.chdir(newdir)
