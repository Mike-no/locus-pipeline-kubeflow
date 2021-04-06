from joblib import load
import numpy as np

class ClusteringModel:

	def __init__(self):
		self._model = load('model.joblib')

	def predict(self, X, feature_names = None, meta = None):
		print(self._model.cluster_centers_.dtype)
		to_predict = np.array([np.array(xi).astype(np.float32) for xi in X])
		print(type(to_predict[0][0]))

		return self._model.predict(to_predict)
