import numpy as np
from metrics import compute_metrics
from channel_selection import build_best_channel_df
from typing import Literal



# -----------------------------
# Analyze missing peaks of the best ibis ch
# -----------------------------

def analyze_missing_peaks(participant: Literal['infant', 'mom'], peaks_data_dict, ibis_data_dict, best_ibis_ch_dict, refined_best_ch=True):
    """
    Analyze missing peaks in the best IBI channel per subject compared to other channels.

    Returns:
    - report: dict keyed by subject, with missing peak info per subject
    """
    report = {}
    
    for subj_id, ibis_subj_data in ibis_data_dict.items():
        peaks_subj_data = peaks_data_dict.get(subj_id, {})
        
        sub_ibi_data = ibis_subj_data.get(participant, {})
        sub_peak_data = peaks_subj_data.get(participant, {})
        
        # Extract ibis channels data
        ibis_channels = {}
        for ch_key, ch_data in sub_ibi_data.items():
            if 'ch' in ch_key and 'data' in ch_data:
                ibis_channels[ch_key] = ch_data['data']
        
        # Extract peaks channels data
        peaks_channels = {}
        for ch_key, ch_data in sub_peak_data.items():
            if 'ch' in ch_key and 'data' in ch_data:
                peaks_channels[ch_key] = ch_data['data']
        
        best_ch = best_ibis_ch_dict.get(subj_id)
        best_ch_ibi_data = ibis_channels.get(best_ch)
        best_ch_peak_data = peaks_channels.get(best_ch)
        
        if best_ch_peak_data is None:
            print(f"Warning: No peak data for  {subj_id} best ch {best_ch} .")

            continue  # skip if no data for best channel

        # Validate IBI matches difference of peaks
        inferred_ibi = np.diff(best_ch_peak_data)
        min_len = min(len(best_ch_ibi_data), len(inferred_ibi))
        if not np.allclose(best_ch_ibi_data[:min_len], inferred_ibi[:min_len], atol=1e-3):
        # if not np.allclose(best_ch_ibi_data[:len(best_ch_ibi_data)], inferred_ibi[:len(inferred_ibi)]):
            print(f"Warning: IBI data for {subj_id} channel {best_ch} does not match peak differences.")
            continue

        missing_peaks = analyze_missing_peaks_intervals(best_ch_peak_data, peaks_channels, best_ch, refined_best_ch=True)
        
        report[subj_id] = missing_peaks
        print(f'sub {subj_id} best {best_ch} md {np.median(best_ch_ibi_data)} th {(np.median(best_ch_ibi_data))*0.5 }: {report[subj_id]}')
    
    return report

def analyze_missing_peaks_intervals(best_ch_peak_data, peaks_channels, best_ch, refined_best_ch=True):
    """
    Identify missing peaks in other channels relative to the best channel.

    Parameters:
    - best_ch_peak_data: list/array of peaks of the best channel
    - peaks_channels: dict of other channels' peaks
    - best_ch: name of best channel
    - refined_best_ch: bool
        If True, only use core peaks (before_start, after_end, important_peaks) without extensions.

    Returns:
    - missing_peaks_report: dict keyed by interval or 'before_start'/'after_end'
    """
    missing_peaks_report = {}
    best_peaks = np.array(best_ch_peak_data)
    best_ibis = np.diff(best_peaks)
    median_ibi = np.median(best_ibis)
    threshold = median_ibi * 0.75

    other_channels = {ch: np.array(pks) for ch, pks in peaks_channels.items() if ch != best_ch}

    # --- Peaks before the first peak ---
    start_time = best_peaks[0]
    for ch_name, ch_peaks in other_channels.items():
        before_start_peaks = ch_peaks[ch_peaks <= start_time - threshold]
        if len(before_start_peaks) > 0:
            peaks_to_add = before_start_peaks if refined_best_ch else \
                sorted(list(set(np.concatenate([[p, ch_peaks[np.where(ch_peaks == p)[0][0] + 1]] 
                                                if np.where(ch_peaks == p)[0][0] < len(ch_peaks)-1 else [p] 
                                                for p in before_start_peaks]))))
            missing_peaks_report.setdefault('before_start', {'best_channel_peaks':[int(start_time)]})[ch_name] = peaks_to_add.astype(int) if not refined_best_ch else list(map(int, peaks_to_add))

    # --- Peaks after the last peak ---
    end_time = best_peaks[-1]
    for ch_name, ch_peaks in other_channels.items():
        after_end_peaks = ch_peaks[ch_peaks >= end_time + threshold]
        if len(after_end_peaks) > 0:
            peaks_to_add = after_end_peaks if refined_best_ch else \
                sorted(list(set(np.concatenate([[ch_peaks[np.where(ch_peaks == p)[0][0] - 1], p] 
                                                if np.where(ch_peaks == p)[0][0] > 0 else [p] 
                                                for p in after_end_peaks]))))
            missing_peaks_report.setdefault('after_end', {'best_channel_peaks':[int(end_time)]})[ch_name] = peaks_to_add.astype(int) if not refined_best_ch else list(map(int, peaks_to_add))

    # --- Missing peaks inside intervals ---
    for ibi_idx in range(len(best_ibis)):
        interval_start = best_peaks[ibi_idx]
        interval_end = best_peaks[ibi_idx + 1]
        missing_per_ch = {}

        for ch_name, ch_peaks in other_channels.items():
            peaks_in_interval = ch_peaks[(ch_peaks > interval_start) & (ch_peaks < interval_end)]
            important_peaks = [pt for pt in peaks_in_interval if (pt - interval_start) > threshold and (interval_end - pt) > threshold]

            if important_peaks:
                if refined_best_ch:
                    extended_peaks = important_peaks
                else:
                    extended_peaks = sorted(list(set(
                        np.concatenate([[ch_peaks[np.where(ch_peaks == pt)[0][0]-1], pt, ch_peaks[np.where(ch_peaks == pt)[0][0]+1]]
                                        if 0 < np.where(ch_peaks == pt)[0][0] < len(ch_peaks)-1 else [pt]
                                        for pt in important_peaks])
                    )))
                missing_per_ch[ch_name] = list(map(int, extended_peaks))

        if missing_per_ch:
            best_peaks_in_interval = [int(p) for p in best_peaks if interval_start <= p <= interval_end]
            missing_peaks_report[ibi_idx] = {
                'best_channel_peaks': best_peaks_in_interval,
                'other_channels': missing_per_ch
            }

    return missing_peaks_report

