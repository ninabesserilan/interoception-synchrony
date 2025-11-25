import pandas as pd
import numpy as np
import copy
import pickle

def apply_gap_filling_to_data_dict(data_dict, factor=2, infant_ibis_th =600, mom_ibis_th = 1000, save_path = None):
    """
    Applies split_long_ibis to each subject’s IBI array and stores:
    
    ['refined_best_channel_data']['ibis_after_interpolation'] = {
        'data': {sub_id: filled_ibi_array},
        'stats': {sub_id: stats_dict}
    }

    Modifies a copy of the data_dict and returns it.
    """

    processed = copy.deepcopy(data_dict)

    for condition, condition_dict in processed.items():
        for participant, part_data in condition_dict.items():

            subs_stat = part_data['refined_best_channel_data']['new_ibis_stats']
            subs_data = part_data['refined_best_channel_data']['new_ibis_data']['data']

            # Containers for transformed IBIs and their stats
            ibis_after_interpolation_data = {}
            ibis_after_interpolation_stats = {}

            # ---- PROCESS EACH SUBJECT ----
            for sub_id, ibi_arr in subs_data.items():

                # Apply gap-filling
                filled_arr = ibis_interpolation(ibi_arr, factor)
                ibis_after_interpolation_data[sub_id] = filled_arr

                # ---- COMPUTE STATS (your logic) ----
                if participant == 'infant':
                    long_ibi_threshold = infant_ibis_th
                else:
                    long_ibi_threshold = mom_ibis_th


                length_best = len(filled_arr)

                median_best = np.median(filled_arr) if length_best > 0 else np.nan
                mean_best = np.mean(filled_arr) if length_best > 0 else np.nan
                sdrr_best = np.std(filled_arr, ddof=1) if length_best > 1 else np.nan
                long_ibi_count_best = (
                    np.sum(filled_arr > long_ibi_threshold) if length_best > 0 else 0
                )

                name_best = subs_stat[sub_id]['best_channel']
                session_lenght_sec = subs_stat[sub_id]['session_lenght_sec']

                ibis_after_interpolation_stats[sub_id] = {
                    'best_channel': name_best,
                    'session_lenght_sec': session_lenght_sec,
                    'length_ibis_ts': length_best,
                    'median': median_best,
                    'mean': mean_best,
                    'sdrr': sdrr_best,
                    'long_ibi_count': long_ibi_count_best
                }

            # ---- Attach results back inside the data structure ----
            part_data['refined_best_channel_data']['ibis_after_interpolation'] = {
                'data': ibis_after_interpolation_data,
                'stats': ibis_after_interpolation_stats
            }
                # ---- Save as pickle if requested ----
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(processed, f)


    return processed


def ibis_interpolation(ibi_arr, factor=2):
    arr = np.asarray(ibi_arr)
    median_ibi = int(np.median(ibi_arr))

    filled = []

    for val in arr:
        if val > 2 * median_ibi:
            # how many full median intervals fit?
            num_reps = int(val // median_ibi)

            # add the repeated median IBIs
            filled.extend([median_ibi] * num_reps)

        else:
            filled.append(val)

    return np.array(filled)






    
