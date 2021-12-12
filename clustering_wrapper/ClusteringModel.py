from joblib import load
import numpy as np
from minio import Minio 
from minio.error import S3Error

class ClusteringModel:

	def __init__(self):
		client = Minio('10.30.8.38:32389', 'minioadmin', 'minioadmin', secure = False)
		try:
			response = client.get_object("clustering", "clustering.joblib")
			with open('clustering.joblib', 'wb') as fd:
				for d in response:
					fd.write(d)
		finally:
			response.close()
			response.release_conn()

		self._model = load('clustering.joblib')

	def predict(self, X, feature_names = None, meta = None):
		print(self._model.cluster_centers_.dtype)
		to_predict = np.array([np.array(xi).astype(np.float32) for xi in X])
		print(type(to_predict[0][0]))

		return self._model.predict(to_predict)
