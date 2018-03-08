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

if not os.path.exists('VTEA_plots'):
    os.mkdir('VTEA_plots')

os.chdir('VTEA_plots')

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Importing the dataset
data = pd.read_csv(currdir + '\\Datasets\\VTEA.csv')
print(data.shape)   # size of data
#print(data)
data.head()
    
# Getting the values
cols = ['Ch1 mean',
        'Ch2_d1 mean',
        'Ch3_d1 mean',
        'Ch4_d1 mean',
        'Ch5_d1 mean',
        'Ch6_d1 mean',
        'Ch7_d1 mean',
        'Ch1 #pixels',
        'Ch1 sum',
        'Ch1 max',
        'Ch1 SD',
        'Ch1 AR',
        'Ch1 F_min',
        'Ch1 F_max',
        'Ch1 mean_th'#,
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
                'Ch7_d1.mean',
                'Ch1..pixels',
                'Ch1.sum',
                'Ch1.max',
                'Ch1.SD',
                'Ch1.AR',
                'Ch1.F_min',
                'Ch1.F_max',
                'Ch1.mean_th'#,
                ##'Ch1.mean_sq',
                ##'Ch1_d0.mean',
                ##'Ch1_d0.sum',
                ##'Ch1_d0.max',
                #'Ch1_d0.SD'#,
                ##'Ch1_d0.mean_th',
                ##'Ch1_d0.mean_sq'
                ]
X = np.zeros(data.shape)
for i in range(len(cols)):
    X[:, i] = data[cols[i]].values
    
for m in range(len(cols)):
    for n in range(m+1, len(cols)):
        if np.array_equal(X[:, m], X[:, n]):
            print(cols[m] + ' (' + str(m) + ') is ' + cols[n] + ' (' + str(n) + ')')           
        else:
            if np.array_equal(X[:, m] ** 2, X[:, n]):
                print(cols[m] + ' (' + str(m) + ') squared is ' + cols[n] + ' (' + str(n) + ')')
            else:
                if np.array_equal(X[:, m], X[:, n] ** 2):
                    print(cols[m] + ' (' + str(m) + ') is ' + cols[n] + ' (' + str(n) + ') squared')
    
if not os.path.exists('7D'):
    os.mkdir('7D')
    
os.chdir('7D')

newdir = os.getcwd()

#print(X)
ctype = data['Object'].values
#print(ctype)

colors = ['b']*len(ctype)
for i in range(len(ctype)):
    if ctype[i] == 'Disease A':
        colors[i] = 'r'
        
#print(colors)

#plot all cells
#if not os.path.exists('data'):
#    os.mkdir('data')
#    
#os.chdir('data')
fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
for m in range(7):  # row of plot grid, corresponds to y-value
    for n in range(7):  # column of plot grid, corresponds to x-value
        if m != n:
            x = X[:, n]
            y = X[:, m]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax[m,n].scatter(x, y, c=z, s=10, edgecolor='')
                
        if m == 6:  # put x-label axis below bottom row
            ax[m,n].set(xlabel=cols[n])

        if n == 0:  # put y-label axis next to first column
            ax[m,n].set(ylabel=cols[m])

plt.savefig('density.png')    # save plot
plt.clf()
plt.close()
        
#os.chdir(newdir)

X1 = np.array([X[j] for j in range(len(X)) if ctype[j] == 'Normal'])
X2 = np.array([X[j] for j in range(len(X)) if ctype[j] == 'Disease A'])

#plot normal cells
#if not os.path.exists('Normal'):
#    os.mkdir('Normal')
#    
#os.chdir('Normal')
fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
for m in range(7):  # row of plot grid, corresponds to y-value
    for n in range(7):  # column of plot grid, corresponds to x-value
        if m != n:
            x = X1[:, n]
            y = X1[:, m]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax[m,n].scatter(x, y, c=z, s=10, edgecolor='')
                
        if m == 6:  # put x-label axis below bottom row
            ax[m,n].set(xlabel=cols[n])

        if n == 0:  # put y-label axis next to first column
            ax[m,n].set(ylabel=cols[m])

plt.savefig('normaldensity.png')    # save plot
plt.clf()
plt.close()

#os.chdir(newdir)
# plot diseased cells
#if not os.path.exists('Disease A'):
#    os.mkdir('Disease A')
#    
#os.chdir('Disease A')
fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
for m in range(7):  # row of plot grid, corresponds to y-value
    for n in range(7):  # column of plot grid, corresponds to x-value
        if m != n:
            x = X2[:, n]
            y = X2[:, m]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax[m,n].scatter(x, y, c=z, s=10, edgecolor='')
                
        if m == 6:  # put x-label axis below bottom row
            ax[m,n].set(xlabel=cols[n])

        if n == 0:  # put y-label axis next to first column
            ax[m,n].set(ylabel=cols[m])

plt.savefig('diseasedensity.png')    # save plot
plt.clf()
plt.close()
    
os.chdir(currdir)