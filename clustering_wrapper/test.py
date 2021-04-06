from joblib import load
import numpy as np
import json
import requests

embeddings = np.load('embeddings.npy')
clustering_model = load('model.joblib')

print(type(embeddings[0][0]))
print(clustering_model.predict(embeddings))
print(clustering_model.cluster_centers_.dtype)

to_predict = embeddings.tolist()
print(type(to_predict[0][0]))

loaded = json.loads(json.dumps(to_predict))
print(type(loaded[0][0]))

request_body = { 'data': { 'ndarray': to_predict } }

response = requests.post('http://localhost:9000/api/v1.0/predictions', json = request_body)