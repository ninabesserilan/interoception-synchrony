from defs_stats_channels import channel_stats_df, num_over_1000
from def_data_availability import data_availability_df, availability_view_by_participant
import pickle
import pandas as pd
from pathlib import Path

### peaks Data Anlysis - 
pickle_path = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
# pickle_path = Path("/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_peaks_data.pkl")

with open(pickle_path, "rb") as f_peaks:
    peaks_data = pickle.load(f_peaks)

peaks_data_status = data_availability_df(peaks_data)
peaks_availability_view = availability_view_by_participant(peaks_data_status, True)

####### 9 months 

### without toys 

peaks_infants_9_without_toys = channel_stats_df(peaks_data, participant='infant', group='9_months', condition='no_toys', with_all_indices= True) 

peaks_moms_9_without_toys = channel_stats_df(peaks_data, participant='mom', group='9_months', condition='no_toys', with_all_indices= True) 

### with toys 
peaks_infants_9_with_toys = channel_stats_df(peaks_data, participant='infant', group='9_months', condition='toys', with_all_indices= True) 
peaks_moms_9_with_toys = channel_stats_df(peaks_data, participant='mom', group='9_months', condition='toys', with_all_indices= True) 


####### 18 months ("book" and "puzzle" instead of "toys")

### without toys 
peaks_infants_18_without_toys = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='no_toys', with_all_indices= True) 
peaks_moms_18_without_toys = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='no_toys',with_all_indices= True) 

### with book 
peaks_infants_18_with_book = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='book', with_all_indices= True) 
peaks_moms_18_with_book = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='book', with_all_indices= True) 

### with puzzle 
peaks_infants_18_with_puzzle = channel_stats_df(peaks_data, participant='infant', group='18_months', condition='puzzle', with_all_indices= True) 
peaks_moms_18_with_puzzle = channel_stats_df(peaks_data, participant='mom', group='18_months', condition='puzzle', with_all_indices= True) 


# Quality index - how many subject in each group have IBIS values over 1000
channels_lengh_peaks = {
    "infants_9_without_toys": peaks_infants_9_without_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "moms_9_without_toys": peaks_moms_9_without_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "infants_9_with_toys": peaks_infants_9_with_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "moms_9_with_toys": peaks_moms_9_with_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "infants_18_without_toys": peaks_infants_18_without_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "moms_18_without_toys": peaks_moms_18_without_toys[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "infants_18_with_book": peaks_infants_18_with_book[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "moms_18_with_book": peaks_moms_18_with_book[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "infants_18_with_puzzle": peaks_infants_18_with_puzzle[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
    "moms_18_with_puzzle": peaks_moms_18_with_puzzle[['ch_0_lenght', 'ch_1_lenght', 'ch_2_lenght']],
}


# with pd.ExcelWriter("all_views.xlsx") as writer:

#     peaks_availability_view.to_excel(writer, sheet_name= "Availability")


#     over_1000_df.to_excel(writer, sheet_name="Above1000")

#     for name, df in datasets.items():
#         # clean sheet name if too long
#         sheet_name = name[:31]  # Excel limit for sheet names
#         df.to_excel(writer, sheet_name=sheet_name)
