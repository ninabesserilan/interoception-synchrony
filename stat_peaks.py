from def_channel_stats import *
from def_data_availability import *

import pickle

### peaks Data Anlysis - 50 dyads ###

with open("peaks_data.pkl", "rb") as f_peaks:
    peaks_data = pickle.load(f_peaks)
peaks_data_status = data_availability_df(peaks_data)
peaks_availability_view = availability_view_by_participant(peaks_data_status)

####### 9 months 

### without toys - 28 infants, 27 moms
peaks_infants_9_without_toys = channel_stats_df(peaks_data, participant='infant', group='9_months', condition='without_toys') 
peaks_moms_9_without_toys = channel_stats_df(peaks_data, participant='mom', group='9_months', condition='without_toys') 

### with toys - 26 infants, 25 moms
peaks_infants_9_with_toys = channel_stats_df(peaks_data, participant='infant', group='9_months', condition='with_toys') 
peaks_moms_9_with_toys = channel_stats_df(peaks_data, participant='mom', group='9_months', condition='with_toys') 


####### 18 months ("book" and "puzzle" instead of "toys")

### without toys - 29 infants, 30 moms
peaks_infants_18_without_toys = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='without_toys') 
peaks_moms_18_without_toys = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='without_toys') 

### with book - 28 infants, 28 moms
peaks_infants_18_with_book = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='with_book') 
peaks_moms_18_with_book = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='with_book') 

### with puzzle - 27 infants, 27 moms
peaks_infants_18_with_puzzle = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='with_puzzle') 
peaks_moms_18_with_puzzle = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='with_puzzle') 
