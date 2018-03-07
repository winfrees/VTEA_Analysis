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
        'Ch7_d1 mean'#,
        #'Ch1 #pixels',
        #'Ch1 sum',
        #'Ch1 max',
        #'Ch1 SD',
        #'Ch1 AR',
        #'Ch1 F_min',
        #'Ch1 F_max',
        #'Ch1 mean_th',
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
                #'Ch1..pixels',
                #'Ch1.sum',
                #'Ch1.max',
                #'Ch1.SD',
                #'Ch1.AR',
                #'Ch1.F_min',
                #'Ch1.F_max',
                #'Ch1.mean_th',
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
for m in range(len(cols)):
    for n in range(len(cols)):
        ax[n,m].scatter(X[:, m], X[:, n], s=10, color = colors)
        if n == 6:
            ax[n,m].set(xlabel=cols[m])
            
        if m == 0:
            ax[n,m].set(ylabel=cols[n])
            
plt.suptitle('Ch1-Ch7 means')
plt.savefig('data.png')
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
for m in range(len(cols)):
    for n in range(len(cols)):
        ax[n,m].scatter(X1[:, m], X1[:, n], s=10, color = 'b')
        if n == 6:
            ax[n,m].set(xlabel=cols[m])
            
        if m == 0:
            ax[n,m].set(ylabel=cols[n])
            
plt.suptitle('Ch1-Ch7 means, normal cells')
plt.savefig('normal.png')
plt.clf()
plt.close()

# plot diseased cells
#if not os.path.exists('Disease A'):
#    os.mkdir('Disease A')
#    
#os.chdir('Disease A')
fig, ax = plt.subplots(7,7, sharex = 'all', sharey = 'all')
for m in range(len(cols)):
    for n in range(len(cols)):
        ax[n,m].scatter(X2[:, m], X2[:, n], s=10, color = 'r')
        if n == 6:
            ax[n,m].set(xlabel=cols[m])
            
        if m == 0:
            ax[n,m].set(ylabel=cols[n])
            
plt.suptitle('Ch1-Ch7 means, Disease A')
plt.savefig('disease.png')
plt.clf()
plt.close()
    
os.chdir(currdir)