from sync import rsa_magnitude, rsa_per_epoch, rsa_time_series, time_series_synchrony, cross_correlation_zlc, multimodal_synchrony
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def prepare_sample_for_analysis(data: dict, min_session_length_sec, missing_ibis_prop=0.20,
):


    sample_for_calculation = {}
    excluded_summary = {}

    for condition, condition_dict in data.items():
        sample_for_calculation[condition] = {}
        excluded_summary[condition] = {}

        for participant, part_data in condition_dict.items():
            subs_stat = part_data['refined_best_channel_data']['new_ibis_stats']
            subs_data = part_data['refined_best_channel_data']['new_ibis_data']['data']

            # Exclude invalid subjects
            excluded_subs = exclude_invalid_subs(
                subs_stat,
                missing_ibis_prop,
                min_session_length_sec
            )

            # Filter out excluded subjects from data
            valid_subs_data = {
                sub_id: data for sub_id, data in subs_data.items()
                if sub_id not in excluded_subs
            }

            # Store valid data for further analysis
            sample_for_calculation[condition][participant] = valid_subs_data

            # Keep track of excluded ones
            excluded_summary[condition][participant] = excluded_subs

    return sample_for_calculation, excluded_summary

    
    


def exclude_invalid_subs(subs_stat: dict, missing_ibis_prop: float, min_session_length_sec:None):
        
        excluded_subs = {}

        if min_session_length_sec == None:
        
            median_session_length = np.median([v['session_lenght_sec'] for v in subs_stat.values()])
            min_session_length = 0.50 * median_session_length
            min_length_criteria = f"half the median length ({min_session_length:.2f}s)"
        
        else:
            min_session_length = min_session_length_sec
            min_length_criteria = f'{min_session_length_sec} sec'

        for sub, stat_data in subs_stat.items():
            if stat_data['session_lenght_sec'] < min_session_length:
                reason = f"Session length ({stat_data['session_lenght_sec']:.2f}s) is shorter than {min_length_criteria}"
                # print(f"Warning: {sub} -> {reason}")
                excluded_subs[sub] = reason
            
        for sub, stat_data in subs_stat.items():
            if sub in excluded_subs:
                continue  # already excluded for another reason
            if stat_data['long_ibi_count'] >= missing_ibis_prop * stat_data['length_ibis_ts']:
                reason = f"Long IBI count ({stat_data['long_ibi_count']}) exceeds {missing_ibis_prop*100:.0f}% of series length ({stat_data['length_ibis_ts']})"
                # print(f"Warning: {sub} -> {reason}")
                excluded_subs[sub] = reason

        return excluded_subs

                  

             
             
        
        



    

