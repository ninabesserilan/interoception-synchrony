from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from def_filter_pickle import filter_group_condition

all_data_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_data.pkl")
ibi_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl") 
peaks_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_peaks_data.pkl")


with open(all_data_pickle, "rb") as f_data:
    all_data = pickle.load(f_data)

with open(ibi_pickle, "rb") as f_ibis:
    ibis_data = pickle.load(f_ibis)

with open(peaks_pickle, "rb") as f_peaks:
    peaks_data = pickle.load(f_peaks)


ibi_toys_9mon_infants_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "infant")
ibi_toys_9mon_moms_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "mom")

ibi_no_toys_9mon_infants_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "infant")
ibi_no_toys_9mon_moms_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "mom")


peaks_toys_9mon_infants_data = filter_group_condition(peaks_data, group="9_months", condition="toys", participant= "infant")
peaks_toys_9mon_moms_data = filter_group_condition(peaks_data, group="9_months", condition="toys", participant= "mom")

peaks_no_toys_9mon_infants_data = filter_group_condition(peaks_data, group="9_months", condition="no_toys", participant= "infant")
peaks_no_toys_9mon_moms_data = filter_group_condition(peaks_data, group="9_months", condition="no_toys", participant= "mom")


# An explanation:

# The ibi is the interval between 2 peaks. 
# So, the first ibi value should be the difference between the second peak and the first peak.
# For example:

example_first_ibi = ibi_toys_9mon_infants_data['01']['infant']['ch_0']['data'][0]
example_first_peak = peaks_toys_9mon_infants_data['01']['infant']['ch_0']['data'][0]
example_second_peak = peaks_toys_9mon_infants_data['01']['infant']['ch_0']['data'][1]

example_first_ibi == example_second_peak - example_first_peak

for idx in range(0, len(ibi_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'])):
    ibi = ibi_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx]
    first_peak = peaks_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx]
    second_peak = peaks_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx+1]
    print(ibi == second_peak - first_peak)

