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
    data_dic = {}

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
            data_dic[subj_id] = row
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

        data_dic[subj_id] = row

    # Build DataFrame
    df = pd.DataFrame(list(data_dic.values()))

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

    return df, data_dic, best_ibis_ch_dict


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


# def build_refined_best_channels_dict(ibis_data_dict, peaks_data_dict, participant):
#     # Build initial best channel info
#     df, data_dic, best_ibis_ch_dict = build_best_channel_df(ibis_data_dict, participant, short_channel_pct=0.9)
    
#     # --- Analyze missing peaks once for all subjects ---
#     missing_peaks_report = analyze_missing_peaks(participant, peaks_data_dict, ibis_data_dict, best_ibis_ch_dict, refined_best_ch=True)

#     final_dict = {}

#     for subj_id, subj_row in data_dic.items():
#         subj_dict = {}
#         best_ch = subj_row.get("best_channel")
#         medium_ch = subj_row.get("medium_channel")
#         worst_ch = subj_row.get("worst_channel")

#         # --- Original channels ---
#         original_channels = {}
#         for label, ch in zip(["best", "medium", "worst"], [best_ch, medium_ch, worst_ch]):
#             if ch is None:
#                 original_channels[label] = {k: np.nan for k in [
#                     "original_peaks", "original_ibis", "original_median", "original_sdrr",
#                     "original_length", "original_long_ibi_count", "original_mean"]}
#                 continue

#             ibis_vals = np.atleast_1d(ibis_data_dict[subj_id][participant][ch]['data']).astype(float)
#             peaks_vals = np.atleast_1d(peaks_data_dict[subj_id][participant][ch]['data']).astype(float)
#             metrics = compute_metrics(ibis_vals, participant)

#             original_channels[label] = {
#                  "name": ch,
#                 "original_peaks": peaks_vals,
#                 "original_ibis": ibis_vals,
#                 "original_median": np.nanmedian(ibis_vals),
#                 "original_sdrr": metrics.get("sdrr", np.nan),
#                 "original_length": len(ibis_vals),
#                 "original_long_ibi_count": metrics.get("long_ibi_count", np.nan),
#                 "original_mean": np.nanmean(ibis_vals)
#             }

#         subj_dict["original_channels"] = original_channels

#         # --- New best channel (refined) ---
#         best_ch_peaks = peaks_data_dict.get(subj_id, {}).get(participant, {}).get(best_ch)
#         if best_ch_peaks is None or subj_id not in missing_peaks_report:
#             subj_dict["new_best_channel"] = {
#                 "name": best_ch,
#                 "peaks": {"data": np.nan, "source": np.nan},
#                 "ibis": np.nan,
#                 "median": np.nan,
#                 "sdrr": np.nan,
#                 "length": np.nan,
#                 "long_ibi_count": np.nan,
#                 "mean": np.nan
#             }
#             final_dict[subj_id] = subj_dict
#             continue

#         best_ch_peaks = np.atleast_1d(best_ch_peaks)
#         new_peaks, sources = insert_missing_peaks_refined(best_ch_peaks, missing_peaks_report[subj_id])
#         new_ibis = np.diff(new_peaks)
#         metrics_new = compute_metrics(new_ibis, participant)

#         subj_dict["new_best_channel"] = {
#             "name": best_ch,
#             "peaks": {"data": new_peaks, "source": sources},
#             "ibis": new_ibis,
#             "median": np.nanmedian(new_ibis),
#             "sdrr": metrics_new.get("sdrr", np.nan),
#             "length": len(new_ibis),
#             "long_ibi_count": metrics_new.get("long_ibi_count", np.nan),
#             "mean": np.nanmean(new_ibis)
#         }

#         final_dict[subj_id] = subj_dict

#     return final_dict


# def insert_missing_peaks_refined(best_ch_peaks, missing_peaks_for_subj):
#     """
#     Insert missing peaks into the best channel using the refined peaks report.
#     Only uses 'before_start', 'after_end', and interval important peaks.
    
#     Returns:
#         new_peaks: sorted array of peaks
#         sources: dict mapping peak -> origin ('best_ch' or channel name)
#     """
#     # If peaks are stored as dict {'data': [...], ...}, extract the numeric array
#     if isinstance(best_ch_peaks, dict) and 'data' in best_ch_peaks:
#         best_ch_peaks = np.atleast_1d(best_ch_peaks['data'])
#     else:
#         best_ch_peaks = np.atleast_1d(best_ch_peaks)

#     new_peaks = list(best_ch_peaks)
#     sources = {int(p): 'best_ch' for p in best_ch_peaks if np.isfinite(p)}

#     for interval_key, interval_data in missing_peaks_for_subj.items():
#         if interval_key in ['before_start', 'after_end']:
#             for ch_name, ch_peaks in interval_data.items():
#                 if ch_name == 'best_channel_peaks':
#                     continue
#                 for p in ch_peaks:
#                     p_int = int(p)
#                     if p_int not in sources:
#                         new_peaks.append(p_int)
#                         sources[p_int] = ch_name
#         else:  # numeric interval
#             other_ch_peaks = interval_data.get('other_channels', {})
#             for ch_name, ch_peaks in other_ch_peaks.items():
#                 for p in ch_peaks:
#                     p_int = int(p)
#                     if p_int not in sources:
#                         new_peaks.append(p_int)
#                         sources[p_int] = ch_name

#     new_peaks = np.sort(np.unique(new_peaks))
#     return new_peaks, sources





# def summarize_missing_peaks(participant, peaks_data_dict, ibis_data_dict, best_ibis_ch_dict, best_df):
#     """
#     Summarize missing peaks per subject into structured dictionaries:
#     {
#         "Peaks before start": {"peaks in medium": x, "peaks in worst": y, "peaks in both": z},
#         "Peaks between ibis": {...},
#         "Peaks after end": {...}
#     }
#     """
#     report = analyze_missing_peaks(participant, peaks_data_dict, ibis_data_dict, best_ibis_ch_dict)
#     summary_rows = []

#     for subj_id, missing_info in report.items():
#         best_ch = best_ibis_ch_dict.get(subj_id)
        
#         # --- find medium and worst channels from best_df ---
#         row_info = best_df[best_df["subject_id"] == subj_id]
#         medium_ch = row_info["medium_channel"].iloc[0] if not row_info.empty else None
#         worst_ch = row_info["worst_channel"].iloc[0] if not row_info.empty else None

#         # Recompute median + threshold
#         ibis_channels = ibis_data_dict[subj_id][participant]
#         best_ch_ibi_data = ibis_channels[best_ch]['data']
#         median_best_ch = np.median(best_ch_ibi_data)
#         threshold = median_best_ch * 0.5

#         # Helper
#         def init_counts():
#             return {"peaks in medium": 0, "peaks in worst": 0, "peaks in both": 0}

#         peaks_before_start = init_counts()
#         peaks_between_ibis = init_counts()
#         peaks_after_end = init_counts()

#         # --- BEFORE START ---
#         if 'before_start' in missing_info:
#             channels_present = set(missing_info['before_start'].keys()) - {'best_channel_peaks'}
#             has_medium = medium_ch in channels_present
#             has_worst = worst_ch in channels_present
#             if has_medium and has_worst:
#                 peaks_before_start["peaks in both"] += 1
#             elif has_medium:
#                 peaks_before_start["peaks in medium"] += 1
#             elif has_worst:
#                 peaks_before_start["peaks in worst"] += 1

#         # --- BETWEEN IBIs ---
#         for k, v in missing_info.items():
#             if isinstance(k, int):
#                 channels_present = set(v.get('other_channels', v.keys())) - {'best_channel_peaks'}
#                 has_medium = medium_ch in channels_present
#                 has_worst = worst_ch in channels_present
#                 if has_medium and has_worst:
#                     peaks_between_ibis["peaks in both"] += 1
#                 elif has_medium:
#                     peaks_between_ibis["peaks in medium"] += 1
#                 elif has_worst:
#                     peaks_between_ibis["peaks in worst"] += 1

#         # --- AFTER END ---
#         if 'after_end' in missing_info:
#             channels_present = set(missing_info['after_end'].keys()) - {'best_channel_peaks'}
#             has_medium = medium_ch in channels_present
#             has_worst = worst_ch in channels_present
#             if has_medium and has_worst:
#                 peaks_after_end["peaks in both"] += 1
#             elif has_medium:
#                 peaks_after_end["peaks in medium"] += 1
#             elif has_worst:
#                 peaks_after_end["peaks in worst"] += 1

#         summary_rows.append({
#             'subject_id': subj_id,
#             'best_channel': best_ch,
#             'median_best_channel': median_best_ch,
#             'threshold': threshold,
#             'Peaks before start': peaks_before_start,
#             'Peaks between ibis': peaks_between_ibis,
#             'Peaks after end': peaks_after_end
#         })

#     df = pd.DataFrame(summary_rows)
#     return df



# # -----------------------------
# # Channel Agreement
# # -----------------------------
# def compute_channel_consistency(ibis_channels):
#     ch_names = list(ibis_channels.keys())
#     if not ch_names:
#         return {}, {}

#     max_len = max(len(np.atleast_1d(ibis_channels[ch])) for ch in ch_names)
#     padded = {
#         ch: np.pad(np.atleast_1d(ibis_channels[ch]).astype(float),
#                    (0, max_len - len(np.atleast_1d(ibis_channels[ch]))),
#                    constant_values=np.nan)
#         for ch in ch_names
#     }

#     pairwise_diffs = {}
#     ch_diffs = {ch: [] for ch in ch_names}

#     for i in range(len(ch_names)):
#         for j in range(i + 1, len(ch_names)):
#             ch_i, ch_j = ch_names[i], ch_names[j]
#             diff = np.nanmean(np.abs(padded[ch_i] - padded[ch_j]))
#             pairwise_diffs[(ch_i, ch_j)] = diff
#             ch_diffs[ch_i].append(diff)
#             ch_diffs[ch_j].append(diff)

#     channel_consistency = {ch: np.nanmean(diffs) if diffs else np.nan
#                            for ch, diffs in ch_diffs.items()}

#     return pairwise_diffs, channel_consistency


