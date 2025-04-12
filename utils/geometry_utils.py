import torch
import numpy as np

def fit_plane(points):
    # Fit a plane to a set of 3D points using SVD
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]
    return {"normal": torch.tensor(normal, device="cuda"), "point": torch.tensor(centroid, device="cuda")}

def detect_planes(points, eps=0.1, min_samples=10):
    # Detect planes using DBSCAN and fit_plane
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = dbscan.labels_
    unique_labels = set(labels)
    planes = []
    for label in unique_labels:
        if label == -1:  # Ignore noise points
            continue
        cluster_points = points[labels == label]
        planes.append(fit_plane(cluster_points))
    return planes

def detect_symmetry(points):
    # Detect symmetry plane using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    normal = pca.components_[-1]
    centroid = points.mean(axis=0)
    return {"normal": torch.tensor(normal, device="cuda"), "point": torch.tensor(centroid, device="cuda")}
