from typing import List
from sync import rsa_magnitude, rsa_per_epoch, rsa_time_series, epochs_synchrony, cross_correlation_zlc, multimodal_synchrony
import numpy as np
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import pickle

def validate_array(arr: List[pd.Series]):
    for val in arr:
        if np.isinf(val) or np.isnan(val):
            print("ERROROROROROR")

def clean_array(arr: List[pd.Series], sub_id, participant, ibi_value_th):
    # Flatten arr in case it contains pd.Series elements
    values = []
    for x in arr:
        if isinstance(x, pd.Series):
            values.extend(x.tolist())
        else:
            values.append(x)

    # Count values above the threshold
    num_above = sum(v >= ibi_value_th for v in values)

    if num_above > 0:
        print(f" {num_above} values >= {ibi_value_th} detected for {sub_id}, {participant}")
    
    return list(filter(lambda n: n < ibi_value_th, arr))


def calculate_rsa(valid_sample:dict,  ibi_value_th:int,require_partner=True):
    """
    Full pipeline to calculate RSA time series for infant-mom pairs.
    
    Parameters
    ----------
    valid_sample : dict
        Structure: valid_sample[condition][participant][subject_id] = time_series
    require_partner : bool, default True
        If True, only calculate RSA for subjects with both participants present (aligned).
        If False, calculate RSA for each participant independently, ignoring unmatched subjects.
    
    Returns
    -------
    new_rsa_dict : dict
        RSA results: new_rsa_dict[condition][subject_id]['infant'] and/or ['mom']
    excluded_summary : dict
        Subjects excluded due to mismatches between participants (empty if require_partner=False)
    """

    excluded_summary = {}
    
    if require_partner:
        # Step 1: Align subjects and exclude unmatched ones
        sample_to_analysis, excluded_summary = exclude_unmatched_pairs(valid_sample)
    else:
        # Skip alignment; use all subjects as-is
        sample_to_analysis = valid_sample

    rsa_dict = {}

    for condition, condition_dict in sample_to_analysis.items():
        rsa_dict[condition] = {}
        # Process each participant type separately
        for p, sub_dict in condition_dict.items():
            for sub_id, ts in sub_dict.items():
                if sub_id not in rsa_dict[condition]:
                    rsa_dict[condition][sub_id] = {}
                age_type = 'infant' if p == 'infant' else 'adult'
                # print(pd.Series(ts))
                
                if ts is None:
                    rsa_dict[condition][sub_id][p] = {}
                else:
                    # ibi_ts = clean_array(list(pd.Series(ts)), sub_id, p, ibi_value_th)
                    ibi_ts = list(pd.Series(ts))
                    validate_array(ibi_ts)
                    rsa_dict[condition][sub_id][p] = rsa_time_series(
                        ibi_ms=ibi_ts,
                        rsa_method='abbney',
                        age_type=age_type
                    )
    return rsa_dict, excluded_summary


def exclude_unmatched_pairs(valid_sample:dict):

    sample_to_analysis = {}
    excluded_summary = {}

    for condition, condition_dict in valid_sample.items():
        sample_to_analysis[condition] = {}
        excluded_summary[condition] = {}

        participants = list(condition_dict.keys())

        # Get subject IDs per participant
        subj_sets = {p: set(condition_dict[p].keys()) for p in participants}

        # Find common and unmatched subjects
        common_subs = set.intersection(*subj_sets.values())
        all_subs = set.union(*subj_sets.values())

        unmatched_subs = all_subs - common_subs

        # Exclude subjects that are missing from one side
        if unmatched_subs:
            excluded_summary[condition]['unmatched_subjects'] = list(unmatched_subs)
            # print(f"Excluding unmatched subjects in condition '{condition}': {unmatched_subs}")

            # Remove them from all participant data
            for p in participants:
                for sub in unmatched_subs:
                    condition_dict[p].pop(sub, None)

        # Initialize sample_to_analysis dics for this condition with cleaned data ---
        sample_to_analysis[condition] = {p: condition_dict[p] for p in participants}


    return sample_to_analysis, excluded_summary





