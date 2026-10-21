"""Spatial niches for the beyond-niche null: k-means on local cell-type composition."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def build_niches(coord, cell_type, n_ct, k_spatial=30, K=10, seed=0):
    nn = NearestNeighbors(n_neighbors=k_spatial + 1).fit(coord)
    _, idx = nn.kneighbors(coord)
    idx = idx[:, 1:]
    comp = np.zeros((len(coord), n_ct), np.float32)
    ct_neigh = cell_type[idx]
    for c in range(n_ct):
        comp[:, c] = (ct_neigh == c).mean(axis=1)
    return MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3).fit(comp).labels_
