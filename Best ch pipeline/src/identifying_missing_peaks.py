import numpy as np
from metrics import compute_metrics
from channel_selection import channel_selection
from typing import Literal



# -----------------------------
# Analyze missing peaks of the best ibis ch
# -----------------------------

def analyze_missing_peaks(participant: Literal['infant', 'mom'], peaks_data_dict, ibis_data_dict, ch_selection_dict, median_ibis_percantage_th =0.75, refined_best_ch=True):
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
        
        best_ch = ch_selection_dict[subj_id]['best_channel']

        best_ch_ibi_data = ibis_channels.get(best_ch)
        best_ch_peak_data = peaks_channels.get(best_ch)
        
        if best_ch_peak_data is None:
            print(f"Warning: No peak data for  {subj_id} best ch {best_ch} .")

            continue  # skip if no data for best channel

        # Validate IBI matches difference of peaks
        inferred_ibi = np.diff(best_ch_peak_data)
        min_len = min(len(best_ch_ibi_data), len(inferred_ibi))
        if not np.allclose(best_ch_ibi_data[:min_len], inferred_ibi[:min_len], atol=1e-3):
            print(f"Warning: IBI data for {subj_id} channel {best_ch} does not match peak differences.")
            continue

        missing_peaks = analyze_missing_peaks_intervals(ch_selection_dict[subj_id], peaks_channels, median_ibis_percantage_th =0.75, refined_best_ch=True)
        
        report[subj_id] = missing_peaks
    
    return report

def analyze_missing_peaks_intervals(sub_ch_selection_dict, peaks_channels, median_ibis_percantage_th =0.75, refined_best_ch=True):
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
    best_ch = sub_ch_selection_dict['best_channel']
    medium_ch = sub_ch_selection_dict['medium_channel']
    worst_ch = sub_ch_selection_dict['worst_channel']

    missing_peaks_report = {}

    best_peaks = np.array(peaks_channels.get(best_ch))
    best_ibis = np.diff(best_peaks)

    median_ibi = sub_ch_selection_dict['median_best']
    threshold = median_ibi * median_ibis_percantage_th

    # Prepare dicts for medium & worst
    other_channels = {
        'medium': {medium_ch: np.array(peaks_channels.get(medium_ch, []))},
        'worst': {worst_ch: np.array(peaks_channels.get(worst_ch, []))}
    }

    # --- Peaks before the first peak ---
    start_time = best_peaks[0]
    for cat, ch_dict in other_channels.items():
        for ch_name, ch_peaks in ch_dict.items():
            before_start_peaks = ch_peaks[ch_peaks <= start_time - threshold]
            if len(before_start_peaks) > 0:
                peaks_to_add = before_start_peaks if refined_best_ch else sorted(
                    list(set(np.concatenate([
                        [p, ch_peaks[np.where(ch_peaks == p)[0][0] + 1]]
                        if np.where(ch_peaks == p)[0][0] < len(ch_peaks) - 1 else [p]
                        for p in before_start_peaks
                    ])))
                )
                missing_peaks_report.setdefault('before_start', {
                    'best_channel_peaks': [int(start_time)],
                    'other_channels': {'medium': {}, 'worst': {}}
                })
                missing_peaks_report['before_start']['other_channels'][cat][ch_name] = list(map(int, peaks_to_add))

    # --- Peaks after the last peak ---
    end_time = best_peaks[-1]
    for cat, ch_dict in other_channels.items():
        for ch_name, ch_peaks in ch_dict.items():
            after_end_peaks = ch_peaks[ch_peaks >= end_time + threshold]
            if len(after_end_peaks) > 0:
                peaks_to_add = after_end_peaks if refined_best_ch else sorted(
                    list(set(np.concatenate([
                        [ch_peaks[np.where(ch_peaks == p)[0][0] - 1], p]
                        if np.where(ch_peaks == p)[0][0] > 0 else [p]
                        for p in after_end_peaks
                    ])))
                )
                missing_peaks_report.setdefault('after_end', {
                    'best_channel_peaks': [int(end_time)],
                    'other_channels': {'medium': {}, 'worst': {}}
                })
                missing_peaks_report['after_end']['other_channels'][cat][ch_name] = list(map(int, peaks_to_add))

    # --- Missing peaks inside intervals ---
    for ibi_idx in range(len(best_ibis)):
        interval_start = best_peaks[ibi_idx]
        interval_end = best_peaks[ibi_idx + 1]
        missing_per_cat = {'medium': {}, 'worst': {}}

        for cat, ch_dict in other_channels.items():
            for ch_name, ch_peaks in ch_dict.items():
                peaks_in_interval = ch_peaks[(ch_peaks > interval_start) & (ch_peaks < interval_end)]
                important_peaks = [
                    pt for pt in peaks_in_interval
                    if (pt - interval_start) > threshold and (interval_end - pt) > threshold
                ]

                if important_peaks:
                    if refined_best_ch:
                        extended_peaks = important_peaks
                    else:
                        extended_peaks = sorted(list(set(
                            np.concatenate([
                                [ch_peaks[np.where(ch_peaks == pt)[0][0] - 1], pt, ch_peaks[np.where(ch_peaks == pt)[0][0] + 1]]
                                if 0 < np.where(ch_peaks == pt)[0][0] < len(ch_peaks) - 1 else [pt]
                                for pt in important_peaks
                            ])
                        )))
                    missing_per_cat[cat][ch_name] = list(map(int, extended_peaks))

        if any(missing_per_cat[cat] for cat in missing_per_cat):
            best_peaks_in_interval = [int(p) for p in best_peaks if interval_start <= p <= interval_end]
            missing_peaks_report[ibi_idx] = {
                'best_channel_peaks': best_peaks_in_interval,
                'other_channels': missing_per_cat
            }

    return missing_peaks_report
