import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from def_pick_ch import *


# pickle_path = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
pickle_path = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

with open(pickle_path, "rb") as f_ibis:
    ibis_data = pickle.load(f_ibis)



ibis_toys_9m_infants_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "infant")
ibis_toys_9m_moms_data = filter_group_condition(ibis_data, group="9_months", condition="toys", participant= "mom")

ibis_no_toys_9m_infants_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "infant")
ibis_no_toys_9m_moms_data = filter_group_condition(ibis_data, group="9_months", condition="no_toys", participant= "mom")

exmp_sub = ibis_toys_9m_infants_data['01']['infant']


# Build DataFrame
df = build_best_channel_df(ibis_toys_9m_infants_data, short_channel_pct=0.90)

# Save to CSV
output_file = "channel_selection_results_new2.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
