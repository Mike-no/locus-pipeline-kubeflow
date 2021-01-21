#%%
from typing import List, Any, Tuple
import tensorflow as tf
import numpy as np
from tqdm import tqdm
#%%
def process_dataframe(filename: str, features: List[str]) -> List[np.ndarray]:
    """
    loads openpflow csv file, containing trajectories.
    returns a list arrays of individual samples containing features.

    Args:
        filename (str): path to csv file
        features (List[str]): list of features to return for training

    Returns:
        List[np.ndarray]: List of np arrays containing individual trajectories.
    """
    import pandas as pd
    data = pd.read_csv(filename, sep=',')
    data.columns=['uid', 'datetime', 'lat', 'lon', 'mode', 'factor']
    groups = [df for _, df in tqdm(data.groupby('uid'), desc=f'grouping dataset')]
    arrays = [df[features].to_numpy() for df in tqdm(groups, desc='generating numpy samples')]
    return arrays

def filter_samples(data: List[np.ndarray], size: int, seqlen: int) -> np.ndarray:
    """
    Filter data to sequences of length `seqlen`.
    Select `size` random trajectories if `size` is positive.
    Basically this function makes it possible to store all data in an ndarray.
    TODO: Properly handle truncation of long series to seqlen.

    Args:
        data (List[np.ndarray]): list of original trajectories
        size (int): number of trajectories to keep
        seqlen (int): series length

    Returns:
        np.ndarray: consolidated array of `size` trajectories of length #seqlen
    """
    data = list(filter(lambda d: len(d)>=seqlen, tqdm(data)))
    data = list(map(lambda d: d[:seqlen], data))
    data = np.array(data, dtype=np.float32)
    indices = np.arange(0, len(data))
    if size > 0:
        indices = np.random.choice(indices, size=size).astype(np.int32)
    return data[indices]

def scale_data(data: np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    Scales the data using a MinMax Scaler.
    Returns the scaled data and the scaler.

    Args:
        data (np.ndarray): Original data

    Returns:
        np.ndarray: MinMax scaled data
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(np.concatenate(data, axis=0))
    shape = data.shape
    data = scaler.transform(data.reshape((-1, shape[-1]))).reshape(shape)
    return data, scaler

def transform_coordinates(data: np.ndarray) -> np.ndarray:
    """
    Transform WG84 coordinates to a metric xy grid.

    Args:
        data (np.ndarray): Original data in WG84

    Returns:
        np.ndarray: Transformed data in EPSG:3857
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    def func(arr):
        arr[:,0], arr[:,1] = transformer.transform(arr[:,0], arr[:,1])
        return arr
    data = np.array(list(map(func, data)))
    return data

def shuffle_trajectories(data: np.ndarray) -> np.ndarray:
    """
    Shuffles the trajectories. Doesn't shuffle timesteps.

    Args:
        data (np.ndarray): Original data

    Returns:
        np.ndarray: Shuffled data
    """
    import random
    indices = list(range(len(data)))
    random.shuffle(indices)
    data = data[indices]
    return data

class SeqGenerator(tf.keras.utils.Sequence):
    """
    A Sequence Generator Class. 
    It generates batches of tracklets for a Seq2Seq encoder, with two inputs and one output.
    """
    def __init__(self, data, lookback, batch_size, shuffle=True, drop_remainder=True):
        self.data = data
        self.lookback = lookback
        self.batch_size = batch_size
        self.indices = np.arange(data.shape[0]*np.ceil((data.shape[1]-lookback+1)).astype(int))
        if shuffle:
            np.random.shuffle(self.indices)
        num_batchs = self.indices.shape[0]//batch_size
        remainder = self.indices[-(len(self.indices) - num_batchs*batch_size):]
        self.indices = self.indices[:num_batchs*batch_size].reshape((num_batchs, batch_size))
        if not drop_remainder:
            self.indices = [elem for elem in self.indices]
            self.indices += [remainder]
        self.window = data.shape[1]-lookback+1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        indices = self.indices[i]
        ind0 = np.ceil(indices//self.window).astype(int).reshape((-1,1))
        ind1 = np.array([np.arange(i%self.window, i%self.window+self.lookback) for i in indices])
        enc = self.data[ind0, ind1, :]
        dec = np.concatenate([np.zeros((len(indices), 1, self.data.shape[-1])), enc[:, 1:, :]], axis=1)
        return (enc, dec), enc

def seq2seq(units: int, n_layers: int, shape: Tuple[int, int]) -> Tuple[Any, Any]:
    """
    Create a simple GRU based seq2seq autoencoder.
    It learns to reconstruct the input.

    Args:
        units (int): number of GRU units per layer
        n_layers (int): nuber of GRU layers
        shape (Tuple[int, int]): Input shape

    Returns:
        (Tuple[Any, Any]): End-to-End Autoencder and a sperate instance of the Encoder
    """

    encoder_inputs = tf.keras.layers.Input(shape=shape, name='encoder_inputs')
    encoder_outputs = encoder_inputs
    for i in range(n_layers):
        encoder_outputs, encoder_states = tf.keras.layers.GRU(
            units,
            return_sequences=False if i==n_layers-1 else True,
            return_state=True, 
            name=f'GRU_encoder_{i}')(encoder_outputs)
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = tf.keras.layers.Input(shape=shape, name='decoder_inputs')
    decoder_outputs = decoder_inputs
    for i in range(n_layers):
        decoder_outputs = tf.keras.layers.GRU(
            units, 
            return_sequences=True,
            return_state=False, 
            name=f'GRU_decoder_{i}')(
                decoder_outputs,
                initial_state=encoder_states if i==0 else None)
    decoder_outputs = tf.keras.layers.Dense(shape[-1])(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_model

def fit(train_gen, val_gen, epochs) -> Any:
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './checkpoint.hdf5',
        onitor='val_ade', 
        save_best_only=True)
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=epochs, 
        callbacks=[earlystop, checkpoint])
    return history

def embed(encoder: Any, data_gen: Any) -> np.ndarray:
    embeddings = encoder.predict(data_gen)
    return embeddings

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
    return y_kmeans

#%%
if __name__ == '__main__':
    filename = 'e1.pq'
    data = process_dataframe(filename, ['lat', 'lon'])

    seqlen = int(np.mean([len(d) for d in data]))
    data = filter_samples(data, size=500, seqlen=seqlen)
    data = transform_coordinates(data)
    data, scaler = scale_data(data)
    data = shuffle_trajectories(data)
    train_gen = SeqGenerator(data[:int(.6*len(data))], 60, 512)
    val_gen = SeqGenerator(data[int(.6*len(data)):int(.8*len(data))], 60, 512)
    test_gen = SeqGenerator(data[int(.8*len(data)):], 60, 512)

    model, encoder = seq2seq(8, 4, (60, 2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    history = fit(train_gen, val_gen, 1000)

    score = model.evaluate(test_gen)

    embeddings = embed(encoder, test_gen)
    np.save('embeddings.npy', embeddings)
    labels = cluster(embeddings, 10)
    np.save('labels.npy', labels)
