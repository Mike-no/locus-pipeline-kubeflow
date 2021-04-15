import numpy as np
import argparse
import sys
import os
from pathlib import Path
from joblib import dump

def cluster(data: np.ndarray, max_clusters: int) -> np.ndarray:
    from kneed import KneeLocator
    from sklearn.cluster import KMeans
    ks = np.arange(0, max_clusters)
    inertias = []
    kmeans = []
    for k in ks:
        # Create a KMeans with k clusters
        print(f"Creating KMeans with {k} clusters")
        model = KMeans(n_clusters=k+1)
        model.fit(data)
        inertias.append(model.inertia_)
        kmeans.append(model)

    kn = KneeLocator(
        ks, 
        inertias, 
        curve='convex', 
        direction='decreasing')
    
    n = kn.knee
    model = kmeans[n-1]
    y_kmeans = model.predict(data)

    return y_kmeans, model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "Path of the .npy embeddings file.")
    parser.add_argument('--Output', type = str, help = "Path of the local file where the clustering model should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 5:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)

    embeddings = np.load(args.Input)

    labels, clustering_model = cluster(embeddings, 10)
    dump(clustering_model, args.Output)