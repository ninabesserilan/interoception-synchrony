import numpy as np
import pandas as pd
from typing import Literal


def compute_metrics(ibis_ms, participant: Literal['mom', 'infant']):
    """
    Compute key IBI and HRV metrics for one channel.
    Returns dictionary with metrics.

        - sdrr: Standard deviation of IBIs (overall variability)
        - long_ibi_count: Number of IBIs > threshold (potential pauses)

    """
    ibis = np.atleast_1d(ibis_ms).astype(float)
    ibis = ibis[~np.isnan(ibis)]
    if len(ibis) < 2:
        return {"sdrr": np.nan, "long_ibi_count": np.nan}

    # Thresholds per participant type
    if participant == 'infant':
        long_ibi_threshold = 600
    else:
        long_ibi_threshold = 1000

    diffs = np.diff(ibis)
    return {
        "sdrr": np.std(ibis, ddof=1),
        "long_ibi_count": np.sum(ibis > long_ibi_threshold),
    }


# -----------------------------
# Channel Agreement
# -----------------------------
def compute_channel_consistency(ibis_channels):
    ch_names = list(ibis_channels.keys())
    if not ch_names:
        return {}, {}

    max_len = max(len(np.atleast_1d(ibis_channels[ch])) for ch in ch_names)
    padded = {
        ch: np.pad(np.atleast_1d(ibis_channels[ch]).astype(float),
                   (0, max_len - len(np.atleast_1d(ibis_channels[ch]))),
                   constant_values=np.nan)
        for ch in ch_names
    }

    pairwise_diffs = {}
    ch_diffs = {ch: [] for ch in ch_names}

    for i in range(len(ch_names)):
        for j in range(i + 1, len(ch_names)):
            ch_i, ch_j = ch_names[i], ch_names[j]
            diff = np.nanmean(np.abs(padded[ch_i] - padded[ch_j]))
            pairwise_diffs[(ch_i, ch_j)] = diff
            ch_diffs[ch_i].append(diff)
            ch_diffs[ch_j].append(diff)

    channel_consistency = {ch: np.nanmean(diffs) if diffs else np.nan
                           for ch, diffs in ch_diffs.items()}

    return pairwise_diffs, channel_consistency


# -----------------------------
# Rank channels using dicts with tie handling
# -----------------------------
def rank_channels(metrics_per_ch, weights):
    """
    Rank channels using weights in a dict-based approach.
    Lower total score = better channel.
    Ties in metric receive equal rank.
    """
    directions = {
        'sdrr': False,           # lower is better
        'long_ibi_count': False, # lower is better
    }

    ranks = {ch: {} for ch in metrics_per_ch.keys()}

    for metric, higher_is_better in directions.items():
        # Sort channels by metric
        ch_vals = [(ch, metrics_per_ch[ch][metric]) for ch in metrics_per_ch if not np.isnan(metrics_per_ch[ch][metric])]
        sorted_chs = sorted(ch_vals, key=lambda x: x[1], reverse=higher_is_better)

        # Tie-aware ranking
        rank_val = 1
        last_val = None
        for idx, (ch, val) in enumerate(sorted_chs):
            if last_val is not None and val != last_val:
                rank_val = idx + 1
            ranks[ch][metric + '_rank'] = rank_val * weights.get(metric, 1.0)
            last_val = val

        # Assign worst rank to channels missing metric or nan
        ranked_chs = {ch for ch, _ in sorted_chs}
        worst_rank = len(sorted_chs) + 1
        for ch in metrics_per_ch.keys():
            if ch not in ranked_chs:
                ranks[ch][metric + '_rank'] = worst_rank * weights.get(metric, 1.0)

    # Compute total rank per channel
    total_ranks = {ch: sum(v.values()) for ch, v in ranks.items()}
    # Handle ties in total rank by choosing longest channel among tied best channels
    min_rank = min(total_ranks.values())
    candidates = [ch for ch, total in total_ranks.items() if total == min_rank]
    if len(candidates) == 1:
        best_ch = candidates[0]
    else:
        # Tie in total rank - pick longest channel
        max_len = -1
        best_ch = None
        for ch in candidates:
            length = len(np.atleast_1d(metrics_per_ch[ch].get('length', np.nan)))
            if np.isnan(length):
                length = 0  # treat nan length as 0
            if length > max_len:
                max_len = length
                best_ch = ch

    return best_ch, ranks, total_ranks


# -----------------------------
# Select best channel
# -----------------------------
def select_best_channel(ibis_channels, participant: Literal['mom', 'infant'],
                        short_channel_pct=0.9, weights=None):
    """
    Select the best channel for a participant, keeping invalid (too short) channels
    but ranking them automatically as the worst.
    Works perfectly when you have 3 channels per subject.
    """
    if weights is None:
        weights = {'sdrr': 1.0, 'long_ibi_count': 2.0}

    # Compute lengths
    n_ibis = {ch: len(np.atleast_1d(ibis)) for ch, ibis in ibis_channels.items()}
    max_len = max(n_ibis.values()) if n_ibis else 0

    # Mark validity
    valid_flags = {ch: (n_ibis[ch] >= short_channel_pct * max_len) for ch in n_ibis}
    all_channels = list(n_ibis.keys())

    # Compute metrics for all channels
    metrics_per_ch = {}
    for ch in all_channels:
        metrics = compute_metrics(ibis_channels[ch], participant)
        metrics['length'] = n_ibis[ch]
        metrics['invalid'] = not valid_flags[ch]
        metrics_per_ch[ch] = metrics

    # Compute consistency (optional)
    _, channel_consistency = compute_channel_consistency(ibis_channels)

    # Rank channels
    best_ch, ranks, total_ranks = rank_channels(metrics_per_ch, weights)

    # -----------------------------
    # Handle invalid channel logic
    # -----------------------------
    invalid_channels = [ch for ch, m in metrics_per_ch.items() if m.get('invalid', False)]
    valid_channels = [ch for ch in metrics_per_ch.keys() if ch not in invalid_channels]

    if len(invalid_channels) == 0:
        # All valid → regular ranking, do nothing special
        pass

    elif len(invalid_channels) == 1:
        # One invalid → it's automatically worst
        worst_ch = invalid_channels[0]
        total_ranks[worst_ch] = max(total_ranks.values()) + 1
        if best_ch == worst_ch:
            # Find valid channel with lowest total rank
            valid_ranks = {ch: total_ranks[ch] for ch in valid_channels if ch in total_ranks}
            if valid_ranks:
                best_ch = min(valid_ranks, key=valid_ranks.get)


    elif len(invalid_channels) == 2:
        # Two invalid → valid is automatically best
        best_ch = valid_channels[0]
        total_ranks[best_ch] = min(total_ranks.values()) - 1

        # Re‑rank the two invalids among themselves
        _, ranks_invalid, total_invalid = rank_channels(
            {ch: metrics_per_ch[ch] for ch in invalid_channels}, weights
        )

        max_rank = max(total_ranks.values())
        for i, (ch, rank_val) in enumerate(sorted(total_invalid.items(), key=lambda x: x[1])):
            total_ranks[ch] = max_rank + i + 1
    return best_ch, {
        "metrics": metrics_per_ch,
        "consistency": channel_consistency,
        "ranks": ranks,
        "total_ranks": total_ranks
    }
    
# -----------------------------
# Build summary DataFrame
# -----------------------------
def build_best_channel_df(ibis_data_dict, participant: Literal['mom', 'infant'], 
                          short_channel_pct=0.9, weights=None):
    """
    Build a summary DataFrame with channels ordered by rank (best → medium → worst),
    and columns ordered by parameter type: length → median → sdrr → long_ibi_count → mean.
    """
    rows = []
    best_ibis_ch_dict = {}


    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})

        # Extract channels
        ibis_channels = {}
        for ch_key in sub_data.keys():
            if 'ch' in ch_key and 'data' in sub_data[ch_key]:
                ibis_channels[ch_key] = sub_data[ch_key]['data']

        # Select best channel + ranks
        best_ch, results = select_best_channel(ibis_channels, participant, short_channel_pct, weights)
        row = {"subject_id": subj_id}


        if results is None:
            row.update({"best_channel": None, "medium_channel": None, "worst_channel": None})
            rows.append(row)
            continue

        # Sort channels by total rank
        sorted_channels = sorted(results["total_ranks"].items(), key=lambda x: x[1])
        channel_order = [ch for ch, _ in sorted_channels]

        # Label top 3
        labels = ["best", "medium", "worst"]
        for i, label in enumerate(labels):
            if i < len(channel_order):
                row[f"{label}_channel"] = channel_order[i]
            else:
                row[f"{label}_channel"] = None

        # Store best channel for this subject
        best_ibis_ch_dict[subj_id] = best_ch

        # Add metrics and length/mean/median for each label
        for i, label in enumerate(labels):
            if i >= len(channel_order):
                continue
            ch = channel_order[i]
            metrics = results["metrics"][ch]
            ibis_vals = np.atleast_1d(ibis_channels[ch]).astype(float)
            row[f"length_{label}"] = len(ibis_vals)
            row[f"median_{label}"] = np.nanmedian(ibis_vals)
            row[f"sdrr_{label}"] = metrics.get("sdrr", np.nan)
            row[f"long_ibi_count_{label}"] = metrics.get("long_ibi_count", np.nan)
            row[f"mean_{label}"] = np.nanmean(ibis_vals)

        rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns as desired
    column_order = ["subject_id", 
                    "best_channel", "medium_channel", "worst_channel",
                    "length_best", "length_medium", "length_worst",
                    "long_ibi_count_best", "long_ibi_count_medium", "long_ibi_count_worst",
                    "sdrr_best", "sdrr_medium", "sdrr_worst",
                    "mean_best", "mean_medium", "mean_worst",
                    "median_best", "median_medium", "median_worst"]
    
    # Keep any extra columns at the end
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[column_order + extra_cols]

    return df, best_ibis_ch_dict


# -----------------------------
# Analyze missing peaks of the best ibis ch
# -----------------------------

def analyze_missing_peaks(participant: Literal['infant', 'mom'], peaks_data_dict, ibis_data_dict, best_ibis_ch_dict):
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

        missing_peaks = analyze_missing_peaks_intervals(best_ch_peak_data, peaks_channels, best_ch)
        
        report[subj_id] = missing_peaks
        print(f'sub {subj_id} best {best_ch} md {np.median(best_ch_ibi_data)} th {(np.median(best_ch_ibi_data))*0.5 }: {report[subj_id]}')
    
    return report


def analyze_missing_peaks_intervals(best_ch_peak_data, peaks_channels, best_ch):
    missing_peaks_report = {}
    best_peaks = np.array(best_ch_peak_data)
    best_ibis = np.diff(best_peaks)
    median_ibi = np.median(best_ibis)
    threshold = median_ibi* 0.5 


    other_channels = {ch: peaks for ch, peaks in peaks_channels.items() if ch != best_ch}

    # Check peaks before the first peak in best channel
    start_time = best_peaks[0]
    for ch_name, ch_peaks in other_channels.items():
        ch_peaks = np.array(ch_peaks)
        before_start_peaks = ch_peaks[(ch_peaks <= start_time - threshold)]

        if len(before_start_peaks) > 0:
            extended_peaks_before_start = []
            for p in before_start_peaks:
                # Add the current peak
                extended_peaks_before_start.append(int(p))

                # Find index of this peak in ch_peaks
                idx = np.where(ch_peaks == p)[0][0]

                # Add the next peak (if it exists)
                if idx < len(ch_peaks) - 1:
                    extended_peaks_before_start.append(int(ch_peaks[idx + 1]))

            # Remove duplicates and sort
            extended_peaks_before_start = sorted(list(set(extended_peaks_before_start)))

            missing_peaks_report.setdefault('before_start', {
                'best_channel_peaks': [int(start_time)]
            })[ch_name] = extended_peaks_before_start

    # Check peaks after the last peak in best channel
    end_time = best_peaks[-1]
    for ch_name, ch_peaks in other_channels.items():
        ch_peaks = np.array(ch_peaks)
        after_end_peaks = ch_peaks[ch_peaks >= end_time + threshold]

        if len(after_end_peaks) > 0:
            extended_peaks_after_end = []

            for p in after_end_peaks:
                # Find index of this peak in the original channel array
                idx = np.where(ch_peaks == p)[0][0]

                # Add the previous peak (if it exists)
                if idx > 0:
                    extended_peaks_after_end.append(int(ch_peaks[idx - 1]))

                # Add the current peak
                extended_peaks_after_end.append(int(p))

            # Remove duplicates and sort
            extended_peaks_after_end = sorted(list(set(extended_peaks_after_end)))

            missing_peaks_report.setdefault('after_end', {
                'best_channel_peaks': [int(end_time)],
            })[ch_name] = extended_peaks_after_end

    # Existing interval checking for missing peaks inside intervals
    for ibi_idx in range(len(best_ibis)):
        interval_start = best_peaks[ibi_idx]
        interval_end = best_peaks[ibi_idx + 1]
        missing_per_ch = {}

        for ch_name, ch_peaks in other_channels.items():
            ch_peaks = np.array(ch_peaks)
            peaks_in_interval = ch_peaks[(ch_peaks > interval_start) & (ch_peaks < interval_end)]

            # Select peaks that are not too close to the edges
            important_peaks = [pt for pt in peaks_in_interval
                            if (pt - interval_start) > threshold and (interval_end - pt) > threshold]

            if important_peaks:
                extended_peaks = []
                for pt in important_peaks:
                    idx = np.where(ch_peaks == pt)[0][0]

                    # Add previous peak (if exists)
                    if idx > 0:
                        extended_peaks.append(int(ch_peaks[idx - 1]))

                    # Add the current peak
                    extended_peaks.append(int(pt))

                    # Add following peak (if exists)
                    if idx < len(ch_peaks) - 1:
                        extended_peaks.append(int(ch_peaks[idx + 1]))

                # Remove duplicates and sort
                extended_peaks = sorted(list(set(extended_peaks)))

                missing_per_ch[ch_name] = extended_peaks

        if missing_per_ch:
            # Gather best channel reference info
            best_peaks_in_interval = [int(p) for p in best_peaks
                                    if interval_start <= p <= interval_end]

            missing_peaks_report[ibi_idx] = {
                'best_channel_peaks': best_peaks_in_interval,
                'other_channels': missing_per_ch
        }
    return missing_peaks_report


def summarize_missing_peaks(participant, peaks_data_dict, ibis_data_dict, best_ibis_ch_dict):
    report = analyze_missing_peaks(participant, peaks_data_dict, ibis_data_dict, best_ibis_ch_dict)
    summary_rows = []

    for subj_id, missing_info in report.items():
        best_ch = best_ibis_ch_dict.get(subj_id)

        # Recompute median and threshold
        ibis_channels = ibis_data_dict[subj_id][participant]
        best_ch_ibi_data = ibis_channels[best_ch]['data']
        median_best_ch = np.median(best_ch_ibi_data)
        threshold = median_best_ch * 0.5

        # Count peaks in each category
        peaks_before_start = len(missing_info.get('before_start', {})) - 1 if 'before_start' in missing_info else 0
        peaks_after_end = len(missing_info.get('after_end', {})) - 1 if 'after_end' in missing_info else 0
        peaks_between_ibis = sum(
            len(v['other_channels']) for k, v in missing_info.items() if isinstance(k, int)
        )

        summary_rows.append({
            'subject_id': subj_id,
            'best_channel': best_ch,
            'median_best_channel': median_best_ch,
            'threshold': threshold,
            'Peaks before start': peaks_before_start,
            'Peaks between ibis': peaks_between_ibis,
            'Peaks after end': peaks_after_end
        })

    df = pd.DataFrame(summary_rows)
    return df
