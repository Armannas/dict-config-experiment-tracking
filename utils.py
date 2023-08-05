import os
import pickle
import pandas as pd
import definitions as defs

def create_dir(dir):
    if not os.path.exists(f"{defs.ROOT_DIR}/{dir}"):
        os.makedirs(f"{defs.ROOT_DIR}/{dir}", exist_ok=True)

def load_pickle(dir_path, file):
    with open(f"{defs.ROOT_DIR}/{dir_path}/{file}.pkl", 'rb') as handle:
        data = pd.read_pickle(handle)
    return data


def save_pickle(dir_path, file, data):
    file_path = f"{defs.ROOT_DIR}/{dir_path}/{file}.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f"{defs.ROOT_DIR}/{dir_path}/{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
