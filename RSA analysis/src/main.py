from sync import rsa_magnitude, rsa_per_epoch, rsa_time_series, epochs_synchrony, cross_correlation_zlc, multimodal_synchrony
import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle

from data_loader import data_dict
from prepare_sample import prepare_sample_for_analysis
from rsa_calculation import calculate_rsa, exclude_unmatched_pairs
from excluded_subs_data import excluded_subs_data


valid_sample, excluded_subs = prepare_sample_for_analysis(data_dict, min_session_length_sec= 60, missing_ibis_prop=0.20)
# a, b= exclude_unmatched_pairs(valid_sample)
rsa_dict, excluded_unmatched_subs = calculate_rsa(valid_sample, require_partner= True, ibi_value_th = 3000)

toys_dyad_num = len(rsa_dict['toys'].keys())      # 68
notoys_dyad_num = len(rsa_dict['no_toys'].keys()) #57

# Building united excluded subs data frame


final_excluded_df_toys_infant,final_excluded_df_toys_mom, final_excluded_df_notoys_infant, final_excluded_df_notoys_mom = excluded_subs_data(excluded_subs, excluded_unmatched_subs, data_dict)

parent_dir = Path(__file__).resolve().parent.parent

output_path_original = parent_dir / "All excluded subs.xlsx"

with pd.ExcelWriter(output_path_original, engine="openpyxl") as writer:
    final_excluded_df_toys_infant.to_excel(writer, sheet_name="Infant_9m_Toys", index=False)
    final_excluded_df_toys_mom.to_excel(writer, sheet_name="Mom_9m_Toys", index=False)
    final_excluded_df_notoys_infant.to_excel(writer, sheet_name="Infant_9m_NoToys", index=False)
    final_excluded_df_notoys_mom.to_excel(writer, sheet_name="Mom_9m_NoToys", index=False)

