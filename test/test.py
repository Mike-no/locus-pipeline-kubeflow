from joblib import load
import numpy as np

clustering = load('model.joblib')
embeddings = np.load('embeddings2.npy')

labels = clustering.predict(embeddings)
for l in labels:
    print(l)
print(labels)