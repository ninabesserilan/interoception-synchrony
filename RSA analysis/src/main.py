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




valid_sample, excluded_subs = prepare_sample_for_analysis(data_dict, min_session_length_sec= 60, missing_ibis_prop=0.20)
# a, b= exclude_unmatched_pairs(valid_sample)
rsa_dict, excluded_summary = calculate_rsa(valid_sample, require_partner= True)
