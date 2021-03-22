import pandas as pd
import numpy as np
import argparse
import sys
import os
from typing import List
from tqdm import tqdm
from pathlib import Path

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
    
    data = pd.read_csv(filename, sep = ',')
    data.columns = ['uid', 'datetime', 'lat', 'lon', 'mode', 'factor']
    groups = [df for _, df in tqdm(data.groupby('uid'), desc = f'grouping dataset')]
    arrays = [df[features].to_numpy() for df in tqdm(groups, desc = 'generating numpy samples')]
    return arrays

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "Path of the local .csv file to be loaded and processed.")
    parser.add_argument('--Output', type = str, help = "Path of the local file where the processed data should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 5:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)

    data = process_dataframe(args.Input, ['lat', 'lon'])
    np.savez(args.Output, *data)

    os.rename(args.Output + '.npz', args.Output)