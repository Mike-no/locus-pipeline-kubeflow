import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model('locusencoder', compile = False)

dataset = tf.data.TFRecordDataset('testdata.tfrecord').map(lambda x: tf.io.parse_tensor(x, tf.float32))
sample = dataset.take(1)
sample = list(sample.as_numpy_iterator())[0]

to_predict = np.reshape(sample, (-1, 60, 2))
print(to_predict)

embeddings = model.predict(to_predict)
print(embeddings)

np.save('embeddings2.npy', embeddings)

tolistarr = to_predict.tolist()
print(type(tolistarr[0][0][0]))

loaded = json.loads(json.dumps(tolistarr))
print(type(loaded[0][0][0]))
print(type(loaded))

to_predict2 = np.array([np.array([np.array(xii).astype(np.float32) for xii in xi]) for xi in loaded])
print(to_predict2)
print(type(to_predict2))
print(type(to_predict2[0][0][0]))
print(to_predict2.shape)

print(model.predict(to_predict2))