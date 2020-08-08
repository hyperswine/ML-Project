"""
Model & analyze potential clusters and 'Nearest-Neighbors'.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary.data_clean2 import clean_data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load up Data
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

df = clean_data(data_features)

# dataset without labels
X1 = df.drop(["key_index", "misc_price"], axis=1)
# dataset without labels and inexpressive features
X2 = df.drop(["key_index", "misc_price", "rom", "selfie_camera_video"], axis=1)
# dataset with output classes though unlabelled for clustering
X3 = X1 = df.drop(["key_index"], axis=1)

"""
K-Means for cluster analysis.
"""

from sklearn.cluster import KMeans

# fit the model
clf = KMeans(n_clusters=5, random_state=0).fit(X1)

# get the centres of each cluster
print(clf.cluster_centers_)

# PCA for numeric (quantitative data = dimensions, display size, cpu clock, ram, battery)
pca = PCA(n_components=2)
pca.fit(X2[])

# K-NN Tree


# Plots
