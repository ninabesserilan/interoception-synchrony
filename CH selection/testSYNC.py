from sync import rsa_magnitude
import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from new_pipline import *

# pickle_path_9m = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
pickle_path_9m = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

with open(pickle_path_9m, "rb") as f_ibis:
    ibis_data_9m = pickle.load(f_ibis)

#  9 months

ibis_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "infant")



ibi_values_ms = ibis_toys_9m_infants_data['01']['infant']['ch_0']['data']
rsa_mag = rsa_magnitude(ibi_ms=ibi_values_ms, rsa_method='porges_bohrer_nk2')
print(f"RSA Magnitude: {rsa_mag}")
