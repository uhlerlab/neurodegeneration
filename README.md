The required packages and versions can be found in requirements.txt.

## 1. Image preprocessing
These notebooks contain codes for intensity normalization, nuclear segmentation, and saving cell image crops based on centroids.

processImgs.ipynb

processImgs_new_imgStats.ipynb

processImgs_new_stitchAndSave.ipynb

## 2. Variational autoencoder training
train_cnnvae.ipynb

## 3. Clustering
This notebook performs the clustering and cluster validation process.

jointClustering/clusterIterate.ipynb

## 4. Leave one patient out cross validation of clinical metadata
Notebooks starting with "train_allmeta_clf" contain codes for cross validation, using cluster proportion, neighborhood counts, or neighborhood compositions.

## 5. Plotting
plot_ig.ipynb - Plots the integrated gradients of the clinical metadata classifications.

plot_imgFeatures_thresh8.ipynb

plot_clusterLocation_stats.ipynb

plot_clusterLocation_thresh8.ipynb
