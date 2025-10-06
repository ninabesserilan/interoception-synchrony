from trial_ibi import ibi_toys_9m_infants_data
import numpy as np
import pandas as pd

# def combine_ch_data(ibis_data_dic, sub_id= '01'):
#     pass

    
#     infants_ch = ibis_data_dic[sub_id]['infant']

#     ibis_channels = {
#             "ch0": infants_ch['ch_0']['data'],
#             "ch1": infants_ch['ch_1']['data'],
#             "ch2": infants_ch['ch_2']['data']
#         }
    

ibis_data_dic = ibi_toys_9m_infants_data['01']['infant']


ibis_channels = {
            "ch0": ibis_data_dic['ch_0']['data'],
            "ch1": ibis_data_dic['ch_1']['data'],
            "ch2": ibis_data_dic['ch_2']['data']
        }
    


# Determine maximum length
max_len = max(len(ibis_channels["ch0"]),
              len(ibis_channels["ch1"]),
              len(ibis_channels["ch2"]))


# Pad shorter arrays with zeros
def pad_to_max(arr, length):
    padded = np.zeros(length)
    padded[:len(arr)] = arr
    return padded

ch0 = pad_to_max(ibis_channels["ch0"], max_len)
ch1 = pad_to_max(ibis_channels["ch1"], max_len)
ch2 = pad_to_max(ibis_channels["ch2"], max_len)

# Compute differences
# np.nan operations automatically yield NaN if any side is NaN
diffs = pd.DataFrame({
    "ch0-ch1": ch0 - ch1,
    "ch0-ch2": ch0 - ch2,
    "ch1-ch2": ch1 - ch2,
    "ch0 values": ibis_channels['ch0'],
    "ch1 values": ibis_channels['ch1'],
    "ch2 values": ibis_channels['ch2']  
})
pd.set_option('display.max_rows', None)
diffs
# # Inspect or summarize
# print(diffs.head())
# print("\nSummary statistics:\n", diffs.describe())




