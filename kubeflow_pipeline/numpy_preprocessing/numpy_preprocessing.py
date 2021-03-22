import numpy as np
import argparse
import sys
import os
from typing import List, Any, Tuple
from tqdm import tqdm
from pathlib import Path

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
    data = list(filter(lambda d: len(d) >= seqlen, tqdm(data)))
    data = list(map(lambda d: d[:seqlen], data))
    data = np.array(data, dtype = np.float32)
    indices = np.arange(0, len(data))
    if size > 0:
        indices = np.random.choice(indices, size = size).astype(np.int32)
    return data[indices]

def transform_coordinates(data: np.ndarray) -> np.ndarray:
    """
    Transform WG84 coordinates to a metric xy grid.

    Args:
        data (np.ndarray): Original data in WG84

    Returns:
        np.ndarray: Transformed data in EPSG:3857
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs(4326, 3857, always_xy = True)
    def func(arr):
        arr[:, 0], arr[:, 1] = transformer.transform(arr[:, 0], arr[:, 1])
        return arr
    data = np.array(list(map(func, data)))
    return data

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
    scaler = MinMaxScaler().fit(np.concatenate(data, axis = 0))
    shape = data.shape
    data = scaler.transform(data.reshape((-1, shape[-1]))).reshape(shape)
    return data, scaler

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "Path of the .npz file to be pre-processed")
    parser.add_argument('--Output', type = str, help = "Path of the local file where rhe pre-processed data should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 5:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)

    loaded_data = np.load(args.Input)
    data = []
    for f in loaded_data.files:
        data.append(loaded_data[f])

    seqlen = int(np.mean([len(d) for d in data]))
    data = filter_samples(data, size = 500, seqlen = seqlen)
    data = transform_coordinates(data)
    data, scaler = scale_data(data)
    data = shuffle_trajectories(data)

    np.save(args.Output, data)

    os.rename(args.Output + '.npy', args.Output)