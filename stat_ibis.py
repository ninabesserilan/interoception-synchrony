from def_channel_stats import *
from def_data_availability import *
from def_num_over_1000 import num_over_1000
import pickle

from stat_peaks import peaks_availability_view

### ibis Data Anlysis - 50 dyads ###

with open("ibis_data.pkl", "rb") as f_ibis:
    ibis_data = pickle.load(f_ibis)

ibis_data_status = data_availability_df(ibis_data)
ibis_availability_view = availability_view_by_participant(ibis_data_status, True)

####### 9 months 

### without toys - 28 infants, 27 moms

ibis_infants_9_without_toys = channel_stats_df(ibis_data, participant='infant', group='9_months', condition='no_toys', with_all_indices= True) 
ibis_moms_9_without_toys = channel_stats_df(ibis_data, participant='mom', group='9_months', condition='no_toys', with_all_indices= True) 

### with toys - 26 infants, 25 moms
ibis_infants_9_with_toys = channel_stats_df(ibis_data, participant='infant', group='9_months', condition='toys', with_all_indices= True) 
ibis_moms_9_with_toys = channel_stats_df(ibis_data, participant='mom', group='9_months', condition='toys', with_all_indices= True) 


####### 18 months ("book" and "puzzle" instead of "toys")

### without toys - 28 infants, 29 moms
ibis_infants_18_without_toys = channel_stats_df(ibis_data, participant='infant', group='18_months', condition='no_toys', with_all_indices= True) 
ibis_moms_18_without_toys = channel_stats_df(ibis_data, participant='mom', group='18_months', condition='no_toys',with_all_indices= True) 

### with book - 28 infants, 28 moms
ibis_infants_18_with_book = channel_stats_df(ibis_data, participant='infant', group='18_months', condition='book', with_all_indices= True) 
ibis_moms_18_with_book = channel_stats_df(ibis_data, participant='mom', group='18_months', condition='book', with_all_indices= True) 

### with puzzle - 27 infants, 27 moms
ibis_infants_18_with_puzzle = channel_stats_df(ibis_data, participant='infant', group='18_months', condition='puzzle', with_all_indices= True) 
ibis_moms_18_with_puzzle = channel_stats_df(ibis_data, participant='mom', group='18_months', condition='puzzle', with_all_indices= True) 


# Quality index - how many subject in each group have IBIS values over 1000
datasets = {
    "ibis_infants_9_without_toys": ibis_infants_9_without_toys,
    "ibis_moms_9_without_toys": ibis_moms_9_without_toys,
    "ibis_infants_9_with_toys": ibis_infants_9_with_toys,
    "ibis_moms_9_with_toys": ibis_moms_9_with_toys,
    "ibis_infants_18_without_toys": ibis_infants_18_without_toys,
    "ibis_moms_18_without_toys": ibis_moms_18_without_toys,
    "ibis_infants_18_with_book": ibis_infants_18_with_book,
    "ibis_moms_18_with_book": ibis_moms_18_with_book,
    "ibis_infants_18_with_puzzle": ibis_infants_18_with_puzzle,
    "ibis_moms_18_with_puzzle": ibis_moms_18_with_puzzle,
}

over_1000_df  = num_over_1000(datasets)


with pd.ExcelWriter("all_views.xlsx") as writer:

    ibis_availability_view.to_excel(writer, sheet_name="Availability")
    over_1000_df.to_excel(writer, sheet_name="Above1000")

    for name, df in datasets.items():
        # clean sheet name if too long
        sheet_name = name[:31]  # Excel limit for sheet names
        df.to_excel(writer, sheet_name=sheet_name)
