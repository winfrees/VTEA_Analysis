# VTEA_Analysis


This repository is for organizing, archiving and following progress of efforts towards developing new approaches of analysis to incorporate into VTEA, https://github.com/icbm-iupui/volumetric-tissue-exploration-analysis.  



For new plots and data, go to results1 folder.
Then go to folder corresponding to desired data set.

Within this folder:
data.png: plot of dataset
density.png: density plot of dataset

To see results of clustering methods, select folder corresponding to number of clusters.
Then select folder corresponding to clustering method.

Within each of these folders:
centroids.txt: list of cluster centroids from all 100 trials combined
allcentroids.png: plot of all cluster centroids from all 100 trials combined
densitycentroids.png: density plot of all cluster centroids

There are no plots corresponding to individual trials in the results1 folder.
They were cut out of the program so that it could finish in a reasonable amount of time.
The results folder (not results1) contains these additional plots (for a separate set of 100 trials).
However, the program took too long, so the folder is unfinished.

Trials are separated into folders. Within each trial folder in results\K1\k=[numclusters]\[clustering_method]:
clusters.txt: cluster info, including the size, color, and centroid of each cluster
data.png: plot of the data, with each point colored according to the corresponding cluster
centroids.png: plot of the cluster centroids
cluster[n].png: plot of the data points in cluster [n]