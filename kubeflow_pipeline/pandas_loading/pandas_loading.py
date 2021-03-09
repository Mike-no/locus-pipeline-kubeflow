import pandas as pd
import numpy as np
import argparse
import sys
from typing import List
from tqdm import tqdm

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
    parser.add_argument('-f', '--Filename', help = ".csv file to be loaded and processed")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    file_path = args.Filename
    if not file_path:
        file_path = 'e1.csv'

    print(file_path)

    data = process_dataframe(file_path, ['lat', 'lon'])
    np.save('/processed_data.npy', data)
