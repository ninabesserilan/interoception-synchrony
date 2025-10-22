import argparse
import json
from pathlib import Path
import pandas as pd
from data_loader import ibis_toys_9m_infants_data, ibis_toys_9m_moms_data, ibis_no_toys_9m_infants_data, ibis_no_toys_9m_moms_data # ibis data- 9 month
from data_loader import peaks_toys_9m_infants_data, peaks_toys_9m_moms_data, peaks_no_toys_9m_infants_data, peaks_no_toys_9m_moms_data # peaks data- 9 month
from channel_selection import build_best_channel_df
import pickle


# ---- best ibis channels ---------------------------

df_infant_9m_toys, dict_data_infant_9m_toys, best_ch_infant_9m_toys = build_best_channel_df(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_toys,dict_data_mom_9m_toys ,best_ch_mom_9m_toys = build_best_channel_df(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.90)

df_infant_9m_no_toys, dict_data_infant_9m_no_toys, best_ch_infant_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.90)
df_mom_9m_no_toys, dict_data_mom_9m_no_toys, best_ch_mom_9m_no_toys = build_best_channel_df(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.90)


