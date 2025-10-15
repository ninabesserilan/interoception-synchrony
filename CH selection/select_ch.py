import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from best_ch_pipline import *

pickle_path_9m = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
# pickle_path_9m = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

with open(pickle_path_9m, "rb") as f_ibis:
    ibis_data_9m = pickle.load(f_ibis)

#  9 months

ibis_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "infant")
ibis_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "mom")

ibis_no_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "infant")
ibis_no_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "mom")



# Build DataFrame
df_infant_9m_toys = build_best_channel_df(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_toys = build_best_channel_df(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.90)

df_infant_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.90)

# Save all DataFrames into one Excel file (each as a separate sheet)
output_path = "best_channels_summary.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_infant_9m_toys.to_excel(writer, sheet_name="Infant_9m_Toys", index=False)
    df_mom_9m_toys.to_excel(writer, sheet_name="Mom_9m_Toys", index=False)
    df_infant_9m_no_toys.to_excel(writer, sheet_name="Infant_9m_NoToys", index=False)
    df_mom_9m_no_toys.to_excel(writer, sheet_name="Mom_9m_NoToys", index=False)



