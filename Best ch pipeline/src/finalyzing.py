import numpy as np
import pandas as pd
from typing import Literal
from generate_refined_channels import fill_missing_peaks
from identifying_missing_peaks import analyze_missing_peaks
import pickle


def create_final_data_dict(
    participant: Literal['infant', 'mom'],
    peaks_data_dict,
    ibis_data_dict,
    ch_selection_dict,
    infant_ibis_th =600, mom_ibis_th = 1000, median_ibis_percantage_th = 0.75
):
    """
    Create a unified dictionary with refined best channel data and original data.
    """
    final_dict = {}

    # --- 1. Refined best channel data ---
    refined_dict = {}


    missing_peaks_dict = analyze_missing_peaks(participant, peaks_data_dict, ibis_data_dict, ch_selection_dict, median_ibis_percantage_th, refined_best_ch= True)
    # 1.1 New peaks data
    new_peaks_dict = fill_missing_peaks(
        participant,
        peaks_data_dict,
        ch_selection_dict,
        missing_peaks_dict,median_ibis_percantage_th = 0.75

    )


    refined_dict['new_peaks_data'] = new_peaks_dict
    # Thresholds per participant type

    if participant == 'infant':
        long_ibi_threshold = infant_ibis_th
    else:
        long_ibi_threshold = mom_ibis_th

    # 1.2 New IBIs data
    new_ibis_data = {'data': {}}
    for subj, peaks_series in new_peaks_dict['data'].items():
        peaks_array = peaks_series.values  # convert Series to array
        ib_is = np.diff(peaks_array)
        new_ibis_data['data'][subj] = ib_is
    refined_dict['new_ibis_data'] = new_ibis_data

    # 1.3 New IBIs statistics
    new_ibis_stats = {}
    for subj, ibis in new_ibis_data['data'].items():

        name_best = ch_selection_dict[subj]['best_channel']
        length_best = len(ibis)
        median_best = np.median(ibis) if length_best > 0 else np.nan
        mean_best = np.mean(ibis) if length_best > 0 else np.nan
        sdrr_best = np.std(ibis, ddof=1) if length_best > 1 else np.nan
        long_ibi_count_best = np.sum(ibis > long_ibi_threshold) if length_best > 0 else 0
        peaks = new_peaks_dict['data'][subj]
        last_peak_ms_best = peaks.iloc[-1] if len(peaks) > 0 else np.nan
        session_lenght_sec = last_peak_ms_best/1000

        new_ibis_stats[subj] = {
            'best_channel': name_best,
            'last_peak_ms':last_peak_ms_best,
            'session_lenght_sec': session_lenght_sec,
            'length_ibis_ts_best': length_best,
            'median_best': median_best,
            'mean_best': mean_best,
            'sdrr_best': sdrr_best,
            'long_ibi_count_best': long_ibi_count_best
        }
    # for subj, peaks in new_peaks_dict['data'].items():
    #     last_peak_ms_best = peaks.iloc[-1]
      


    refined_dict['new_ibis_stats'] = new_ibis_stats

    final_dict['refined_best_channel_data'] = refined_dict


    df_new_ibis_stats = pd.DataFrame.from_dict(
        refined_dict['new_ibis_stats'], orient='index'
    ).reset_index().rename(columns={'index': 'subject_id'})

    # Reorder columns as desired
    column_order = ["subject_id", "best_channel", "session_lenght_sec", "last_peak_ms", "length_ibis_ts_best", "long_ibi_count_best", 
                    "sdrr_best",  "mean_best", "median_best"]

    # Keep any extra columns at the end
    extra_cols = [c for c in df_new_ibis_stats.columns if c not in column_order]
    df_new_ibis_stats = df_new_ibis_stats[column_order + extra_cols]

    # --- 2. Original data all channels ---
    final_dict['original_data_all_channels'] = {
        'peaks_data': peaks_data_dict,
        'ibis_data': ibis_data_dict,
        'ibis_stats': ch_selection_dict
    }


    return final_dict, df_new_ibis_stats



def save_all_final_data_pickle(
    toys_infant_final_dict,
    toys_mom_final_dict,
    no_toys_infant_final_dict,
    no_toys_mom_final_dict,
    output_path: str = "all_final_data.pkl"
):
    """
    Assemble all participant/condition final_dicts into a nested dict and save as a pickle.

    Parameters:
    - toys_infant_final_dict: final_dict for infant, toys
    - toys_mom_final_dict: final_dict for mom, toys
    - no_toys_infant_final_dict: final_dict for infant, no_toys
    - no_toys_mom_final_dict: final_dict for mom, no_toys
    - output_path: str, path to save pickle
    """
    all_final_data = {
        'toys': {
            'infant': toys_infant_final_dict,
            'mom': toys_mom_final_dict
        },
        'no_toys': {
            'infant': no_toys_infant_final_dict,
            'mom': no_toys_mom_final_dict
        }
    }

    # Save pickle
    with open(output_path, "wb") as f:
        pickle.dump(all_final_data, f)

    print(f"All data saved to {output_path}")
    return all_final_data
