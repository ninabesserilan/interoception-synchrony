import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from pick_ch import *


pickle_path = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")

with open(pickle_path, "rb") as f_ibis:
    ibis_data = pickle.load(f_ibis)



peaks_toys_9m_infants_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "infant")
peaks_toys_9m_moms_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "mom")

peaks_no_toys_9m_infants_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "infant")
peaks_no_toys_9m_moms_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "mom")

exmp_sub = peaks_toys_9m_infants_data['01']['infant']


# Build DataFrame
df = build_best_channel_df(peaks_toys_9m_infants_data, short_channel_pct=0.95)

# Save to CSV
output_file = "channel_selection_results.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
