import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from best_ch_pipline import *

# ---- Load Pickles ---------------------------

# pickle_path_9m_ibis = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
pickle_path_9m_ibis = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

# pickle_path_9_month_peaks = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_peaks_data.pkl")
pickle_path_9_month_peaks = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_peaks_data.pkl')

with open(pickle_path_9m_ibis, "rb") as f_ibis:
    ibis_data_9m = pickle.load(f_ibis)

with open(pickle_path_9_month_peaks, "rb") as f_peaks:
    peaks_data_9m = pickle.load(f_peaks)

# ---- Data Structure ---------------------------

# 9 months - ibis
ibis_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "infant")
ibis_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "mom")

ibis_no_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "infant")
ibis_no_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "mom")

# 9 months - peaks
peaks_toys_9m_infants_data = filter_group_condition(peaks_data_9m, group="9_months", condition="toys", participant= "infant")
peaks_toys_9m_moms_data = filter_group_condition(peaks_data_9m, group="9_months", condition="toys", participant= "mom")

peaks_no_toys_9m_infants_data = filter_group_condition(peaks_data_9m, group="9_months", condition="no_toys", participant= "infant")
peaks_no_toys_9m_moms_data = filter_group_condition(peaks_data_9m, group="9_months", condition="no_toys", participant= "mom")

# ---- Build df - best ibis channels ---------------------------

df_infant_9m_toys, dict_data_infant_9m_toys, best_ch_infant_9m_toys = build_best_channel_df(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_toys,dict_data_mom_9m_toys ,best_ch_mom_9m_toys = build_best_channel_df(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.90)

df_infant_9m_no_toys, dict_data_infant_9m_no_toys, best_ch_infant_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_no_toys, dict_data_mom_9m_no_toys, best_ch_mom_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.90)

# ---- Analyze missing peaks for best ibis channels ---------------------------
# infant_9m_toys_new_best_ch = compute_best_channels_with_missing(ibis_toys_9m_infants_data, peaks_toys_9m_infants_data, 'infant')

# infant_9m_toys_new_best_ch = analyze_missing_peaks('infant', peaks_toys_9m_infants_data, ibis_toys_9m_infants_data,best_ch_infant_9m_toys, True)

# infant_9m_toys_new_best_ch = build_refined_best_channels_dict(ibis_toys_9m_infants_data, peaks_toys_9m_infants_data, 'infant')
# Save all DataFrames into one Excel file (each as a separate sheet)
output_path = "best_channels_summary.xlsx"

# with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
#     df_infant_9m_toys.to_excel(writer, sheet_name="Infant_9m_Toys", index=False)
#     df_mom_9m_toys.to_excel(writer, sheet_name="Mom_9m_Toys", index=False)
#     df_infant_9m_no_toys.to_excel(writer, sheet_name="Infant_9m_NoToys", index=False)
#     df_mom_9m_no_toys.to_excel(writer, sheet_name="Mom_9m_NoToys", index=False)



# Example usage:
# df_results_infant_9m_toys = summarize_missing_peaks('infant', peaks_toys_9m_infants_data, ibis_toys_9m_infants_data, best_ch_infant_9m_toys,df_infant_9m_toys)

# df_results_infant_9m_toys.to_csv('missing_peaks_summary.csv', index=False)


# df_results_infant_9m_toys_old = summarize_missing_peaks_old('infant', peaks_toys_9m_infants_data, ibis_toys_9m_infants_data, best_ch_infant_9m_toys)

# df_results_infant_9m_toys_old.to_csv('missing_peaks_summary_old.csv', index=False)


# sub7_ch2_best = peaks_toys_9m_infants_data['07']['infant']['ch_2']
# sub7_ch1_worst = peaks_toys_9m_infants_data['07']['infant']['ch_1']
# sub7_ch0_medium = peaks_toys_9m_infants_data['07']['infant']['ch_0']

# df_try = pd.concat(
#     [sub7_ch2_best['data'], sub7_ch0_medium['data'], sub7_ch1_worst['data']],
#     axis=1,
#     ignore_index=False
# )

# # Rename columns
# df_try.columns = ['ch2_best', 'ch0_medium', 'ch1_worst']
