from sync import rsa_magnitude, rsa_per_epoch, rsa_time_series, epochs_synchrony, cross_correlation_zlc, multimodal_synchrony
import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle


# ---- Load Pickle ---------------------------

pickle_path = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Best ch pipeline/all data improved and original chs.pkl')


with open(pickle_path, "rb") as l_data:
    all_data = pickle.load(l_data)



