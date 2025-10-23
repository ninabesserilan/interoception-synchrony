import argparse
import json
from pathlib import Path
import pandas as pd
from data_loader import ibis_toys_9m_infants_data, ibis_toys_9m_moms_data, ibis_no_toys_9m_infants_data, ibis_no_toys_9m_moms_data # ibis data- 9 month
from data_loader import peaks_toys_9m_infants_data, peaks_toys_9m_moms_data, peaks_no_toys_9m_infants_data, peaks_no_toys_9m_moms_data # peaks data- 9 month
from channel_selection import channel_selection
from identifying_missing_peaks import analyze_missing_peaks
from generate_refined_channels import fill_missing_peaks
from finalyzing import create_final_data_dict, save_all_final_data_pickle
import pickle
import openpyxl


# ---- best ibis channels ---------------------------

df_ch_selection_infant_9m_toys, ch_selection_dict_infant_9m_toys = channel_selection(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)
df_ch_selection_mom_9m_toys,ch_selection_dict_mom_9m_toys = channel_selection(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)

df_ch_selection_infant_9m_no_toys, ch_selection_dict_infant_9m_no_toys = channel_selection(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)
df_ch_selection_mom_9m_no_toys, ch_selection_dict_mom_9m_no_toys = channel_selection(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.85,infant_ibis_th =600, mom_ibis_th = 1000)

# save ststs df

output_path_original = "original_best_channels_summary.xlsx"

with pd.ExcelWriter(output_path_original, engine="openpyxl") as writer:
    df_ch_selection_infant_9m_toys.to_excel(writer, sheet_name="Infant_9m_Toys", index=False)
    df_ch_selection_mom_9m_toys.to_excel(writer, sheet_name="Mom_9m_Toys", index=False)
    df_ch_selection_infant_9m_no_toys.to_excel(writer, sheet_name="Infant_9m_NoToys", index=False)
    df_ch_selection_mom_9m_no_toys.to_excel(writer, sheet_name="Mom_9m_NoToys", index=False)


# ---- Insert missing peaks for best ibis channels and creat final dict with original and improved data---------------------------

toys_infant_final_dict, toys_infant_df_new_ibis_stats = create_final_data_dict('infant',peaks_toys_9m_infants_data, ibis_toys_9m_infants_data, ch_selection_dict_infant_9m_toys,infant_ibis_th =600, mom_ibis_th = 1000, median_ibis_percantage_th = 0.75 )
toys_mom_final_dict, toys_mom_df_new_ibis_stats = create_final_data_dict('mom',peaks_toys_9m_moms_data, ibis_toys_9m_moms_data, ch_selection_dict_mom_9m_toys,infant_ibis_th =600, mom_ibis_th = 1000, median_ibis_percantage_th = 0.75 )

notoys_infant_final_dict, notoys_infant_df_new_ibis_stats = create_final_data_dict('infant',peaks_no_toys_9m_infants_data, ibis_no_toys_9m_infants_data, ch_selection_dict_infant_9m_no_toys,infant_ibis_th =600, mom_ibis_th = 1000, median_ibis_percantage_th = 0.75 )
notoys_mom_final_dict, notoys_infants_df_new_ibis_stats = create_final_data_dict('mom',peaks_no_toys_9m_moms_data, ibis_no_toys_9m_moms_data, ch_selection_dict_mom_9m_no_toys,infant_ibis_th =600, mom_ibis_th = 1000, median_ibis_percantage_th = 0.75 )


# save ststs df
output_path_new = "new_best_channels_summary.xlsx"

with pd.ExcelWriter(output_path_new, engine="openpyxl") as writer:
    toys_infant_df_new_ibis_stats.to_excel(writer, sheet_name="Infant_9m_Toys", index=False)
    toys_mom_df_new_ibis_stats.to_excel(writer, sheet_name="Mom_9m_Toys", index=False)
    notoys_infant_df_new_ibis_stats.to_excel(writer, sheet_name="Infant_9m_NoToys", index=False)
    notoys_infants_df_new_ibis_stats.to_excel(writer, sheet_name="Mom_9m_NoToys", index=False)


# save all data to a new pickle
all_data_pickle_output_path = 'all data improved and original chs.pkl'
new_improved_channel_data = save_all_final_data_pickle(toys_infant_final_dict, toys_mom_final_dict, notoys_infant_final_dict, notoys_mom_final_dict, all_data_pickle_output_path)


# ---- Analyze missing peaks for best ibis channels ---------------------------

# toys_infant_9m_missing_peaks = analyze_missing_peaks('infant', peaks_toys_9m_infants_data, ibis_toys_9m_infants_data, ch_selection_dict_infant_9m_toys, median_ibis_percantage_th = 0.75, refined_best_ch=True)
# toys_mom_9m_missing_peaks = analyze_missing_peaks('mom', peaks_toys_9m_moms_data, ibis_toys_9m_moms_data, ch_selection_dict_mom_9m_toys, median_ibis_percantage_th = 0.75, refined_best_ch=True)

# notoys_infant_9m_missing_peaks = analyze_missing_peaks('infant', peaks_no_toys_9m_infants_data, ibis_no_toys_9m_infants_data, ch_selection_dict_infant_9m_no_toys, median_ibis_percantage_th = 0.75, refined_best_ch=True)
# notoys_mom_9m_missing_peaks = analyze_missing_peaks('mom', peaks_no_toys_9m_moms_data, ibis_no_toys_9m_moms_data, ch_selection_dict_mom_9m_no_toys, median_ibis_percantage_th = 0.75, refined_best_ch=True)


