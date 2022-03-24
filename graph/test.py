import tensorflow as tf
import numpy as np
import json
import requests

dataset = tf.data.TFRecordDataset('testdata.tfrecord').map(lambda x: tf.io.parse_tensor(x, tf.float32))
sample = dataset.take(1)
sample = list(sample.as_numpy_iterator())[0]
print(type(sample[0][0][0]))

sample = np.reshape(sample, (-1, 60, 2))

to_predict = sample.tolist()
print(type(to_predict[0][0][0]))

loaded = json.loads(json.dumps(to_predict))
print(type(loaded[0][0][0]))

request_body = { 'data': { 'ndarray': to_predict } }

response = requests.post('http://10.30.5.20:32353/api/v1.0/predictions', json = request_body)
print(response.status_code, response.text)