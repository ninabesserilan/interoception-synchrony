from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from def_filter_pickle import filter_group_condition

# all_data_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_data.pkl")
# ibi_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl") 
# peaks_pickle = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_peaks_data.pkl")


all_data_pickle = Path("Q:\\hoehl\\projects\\mibo_p\\Physiological_Synchrony\\analyses\\Files processing - availability information\\01_ibi_after_extraction_data.pkl")
ibi_pickle = Path("Q:\\hoehl\\projects\\mibo_p\\Physiological_Synchrony\\analyses\\Files processing - availability information\\01_ibi_after_extraction_ibis_data.pkl") 
peaks_pickle = Path("Q:\\hoehl\\projects\\mibo_p\\Physiological_Synchrony\\analyses\\Files processing - availability information\\01_ibi_after_extraction_peaks_data.pkl")


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

# def verify_ibis_vs_peaks(ibi_data, peaks_data):
#     results = {}

#     for subj_id, subj_dict in ibi_data.items():  # loop over subjects
#         results[subj_id] = {}
#         for participant, part_dict in subj_dict.items():  # infant / mom
#             results[subj_id][participant] = {}
#             for ch_name, ch_dict in part_dict.items():  # ch_0, ch_1...
#                 ibis = ch_dict["data"]
#                 peaks = peaks_data[subj_id][participant][ch_name]["data"]

#                 # check all consecutive pairs
#                 comparisons = []
#                 for idx in range(len(ibis)):
#                     if idx + 1 < len(peaks):  # safe check
#                         comparisons.append(ibis[idx] == peaks[idx+1] - peaks[idx])
#                     else:
#                         comparisons.append(None)  # last element has no "next peak"

#                 results[subj_id][participant][ch_name] = all(
#                     c is True for c in comparisons if c is not None
#                 )

#     return results


# # Run it for toys / no_toys, infants / moms, 9_months
# verification_toys_infants = verify_ibis_vs_peaks(
#     ibi_toys_9mon_infants_data, peaks_toys_9mon_infants_data
# )
# verification_toys_moms = verify_ibis_vs_peaks(
#     ibi_toys_9mon_moms_data, peaks_toys_9mon_moms_data
# )
# verification_no_toys_infants = verify_ibis_vs_peaks(
#     ibi_no_toys_9mon_infants_data, peaks_no_toys_9mon_infants_data
# )
# verification_no_toys_moms = verify_ibis_vs_peaks(
#     ibi_no_toys_9mon_moms_data, peaks_no_toys_9mon_moms_data
# )


# An explanation:

# The ibi is the interval between 2 peaks. 
# So, the first ibi value should be the difference between the second peak and the first peak.
# For example:

# example_first_ibi = ibi_toys_9mon_infants_data['01']['infant']['ch_0']['data'][0]
# example_first_peak = peaks_toys_9mon_infants_data['01']['infant']['ch_0']['data'][0]
# example_second_peak = peaks_toys_9mon_infants_data['01']['infant']['ch_0']['data'][1]

# example_first_ibi == example_second_peak - example_first_peak

# for idx in range(0, len(ibi_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'])):
#     ibi = ibi_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx]
#     first_peak = peaks_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx]
#     second_peak = peaks_no_toys_9mon_infants_data['89']['infant']['ch_0']['data'][idx+1]
#     print(ibi == second_peak - first_peak)

