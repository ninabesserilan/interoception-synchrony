import pickle
from pathlib import Path
from typing import Literal


def data_loader(pickle_path: Path):
    with open(pickle_path, "rb") as l_data:
        pickle_data = pickle.load(l_data)
    
    return pickle_data




pickle_path = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Best ch pipeline/all data improved and original chs.pkl')

data_dict = data_loader(pickle_path)