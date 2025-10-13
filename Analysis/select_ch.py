import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle
from def_filter_pickle import filter_group_condition
from ch_selection_pipline import *


# pickle_path_9m = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
pickle_path_9m = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

with open(pickle_path_9m, "rb") as f_ibis:
    ibis_data_9m = pickle.load(f_ibis)

#  9 months

ibis_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "infant")
ibis_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "mom")

ibis_no_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "infant")
ibis_no_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "mom")



# Build DataFrame
# df_infant_9m_toys = build_best_channel_df(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.90)
# df_mom_9m_toys = build_best_channel_df(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.90)

# df_infant_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.90)
# df_mom_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.90)

# Save to CSV


# store all data per channel
# assuming your dict is named data_dict
channel_data = {}

for participant_id, participant_data in ibis_no_toys_9m_moms_data.items():
    infant_data = participant_data.get('mom', {})
    for ch, ch_info in infant_data.items():
        data = ch_info.get('data')

        # ensure the data is a Series
        if isinstance(data, pd.Series):
            s = data
        elif np.isscalar(data):
            s = pd.Series([data])
        elif isinstance(data, (list, np.ndarray)):
            s = pd.Series(data)
        else:
            continue  # skip invalid entries

        channel_data.setdefault(ch, []).append(s)

# compute mean per channel across participants
channel_means = {}
for ch, series_list in channel_data.items():
    if not series_list:
        continue
    # align series of different lengths, ignore missing values
    df = pd.concat(series_list, axis=1, ignore_index=True)
    channel_means[ch] = df.stack().mean()

# display results
for ch, avg in channel_means.items():
    print(f"{ch}: {avg:.2f}")
