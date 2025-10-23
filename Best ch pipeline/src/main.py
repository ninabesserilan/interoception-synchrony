import argparse
import json
from pathlib import Path
import pandas as pd
from data_loader import ibis_toys_9m_infants_data, ibis_toys_9m_moms_data, ibis_no_toys_9m_infants_data, ibis_no_toys_9m_moms_data # ibis data- 9 month
from data_loader import peaks_toys_9m_infants_data, peaks_toys_9m_moms_data, peaks_no_toys_9m_infants_data, peaks_no_toys_9m_moms_data # peaks data- 9 month
from channel_selection import channel_selection
from identifying_missing_peaks import analyze_missing_peaks
from generate_refined_channels import fill_missing_peaks
import pickle


# ---- best ibis channels ---------------------------

df_ch_selection_infant_9m_toys, ch_selection_dict_infant_9m_toys = channel_selection(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)
df_ch_selection_mom_9m_toys,ch_selection_dict_mom_9m_toys = channel_selection(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)

df_ch_selection_infant_9m_no_toys, ch_selection_dict_infant_9m_no_toys = channel_selection(ibis_no_toys_9m_infants_data, "infant", short_channel_pct=0.85, infant_ibis_th =600, mom_ibis_th = 1000)
df_ch_selection_mom_9m_no_toys, ch_selection_dict_mom_9m_no_toys = channel_selection(ibis_no_toys_9m_moms_data, "mom", short_channel_pct=0.85,infant_ibis_th =600, mom_ibis_th = 1000)


# ---- Analyze missing peaks for best ibis channels ---------------------------

infant_9m_toys_missing_peaks = analyze_missing_peaks('infant', peaks_toys_9m_infants_data, ibis_toys_9m_infants_data, ch_selection_dict_infant_9m_toys, median_ibis_percantage_th = 0.75, refined_best_ch=True)


a= fill_missing_peaks('infant', peaks_toys_9m_infants_data, ch_selection_dict_infant_9m_toys, infant_9m_toys_missing_peaks, median_ibis_percantage_th = 0.75)