# import numpy as np
# import pandas as pd
# # -----------------------------
# # Metric computation
# # -----------------------------
# def compute_metrics(ibis_ms, long_ibi_threshold=1000):
#     """
#     Compute standard and HRV metrics for a single IBI vector (ms)



#     Returns
#     -------
#         dict
#         Dictionary containing data quality and HRV metrics:
#         - sdrr: standard deviation of IBIs (overall variability)
#         - long_ibi_threshold: number of IBIs > long_ibi_threshold
#         - rmssd: root mean square of successive differences
#         - pNN50: percentage of successive RR intervals that differ by more than 50 ms

#             Shaffer & Ginsberg, 2017
#             https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/


#     """
#     ibis = np.array(ibis_ms)
#     ibis = ibis[~np.isnan(ibis)]
#     if len(ibis) < 2:
#         return {
#             "sdrr": np.nan,
#             "long_ibi_threshold": np.nan,
#             "rmssd": np.nan,
#             "pnn50": np.nan}

#     std_ibi = np.std(ibis, ddof=1)
#     long_ibi = np.sum(ibis > long_ibi_threshold)

#     # HRV metrics
#     diffs = np.diff(ibis)
#     rmssd = np.sqrt(np.mean(diffs**2))
#     nn50 = np.sum(np.abs(diffs) > 50)
#     pnn50 = nn50 / len(diffs) * 100


#     return {
#         "sdrr": std_ibi,
#         "long_ibi_threshold": long_ibi,
#         "rmssd": rmssd,
#         "pnn50": pnn50
#                 }

# # -----------------------------
# # Channel selection pipeline
# # -----------------------------
# def select_best_channel(ibis_channels, short_channel_pct=0.95):
#     """
#     Select the best ECG channel based on IBI metrics and ranking.
#     Median IBI is NOT included anymore.
#     """
#     metrics_per_ch = {}
#     n_beats = {ch: len(ibis) for ch, ibis in ibis_channels.items()}
#     max_len = max(n_beats.values())
    
#     # Filter out too-short channels
#     valid_channels = [ch for ch, length in n_beats.items() if length >= short_channel_pct * max_len]
#     if not valid_channels:
#         return None, None
    
#     # Calculate metrics
#     for ch in valid_channels:
#         metrics_per_ch[ch] = compute_metrics(ibis_channels[ch])
    
#     # Rank channels
#     ranks_per_ch = {ch: {} for ch in metrics_per_ch}
#     total_scores = {ch: 0 for ch in metrics_per_ch}
    
#     for metric in ["sdrr", "long_ibi_threshold", "rmssd", "pnn50"]:
#         smaller_is_better = metric in ["sdrr", "long_ibi_threshold"]
#         values = {ch: metrics_per_ch[ch][metric] for ch in metrics_per_ch}
#         sorted_chs = sorted(values, key=lambda ch: values[ch], reverse=not smaller_is_better)
        
#         for rank, ch in enumerate(sorted_chs, start=1):
#             ranks_per_ch[ch][metric] = rank
#             total_scores[ch] += rank
    
#     # Best channel = lowest total score
#     best_ch = min(total_scores, key=total_scores.get)
    
#     results = {
#         "metrics": metrics_per_ch,
#         "ranks": ranks_per_ch,
#         "total_scores": total_scores
#     }
#     return best_ch, results

# # DataFrame builder
# # ==========================
# def build_best_channel_df(ibis_data_dict, short_channel_pct=0.95):
#     """
#     data_dict: your dict
#     returns: pandas DataFrame with metrics, ranks, scores, and best_channel_IBIs
#     """
#     rows = []
#     expected_metrics = ["sdrr", "long_ibi_threshold", "rmssd", "pnn50"]

#     for subj_id, subj_data in ibis_data_dict.items():
#         infant_data = subj_data.get('infant', {})
#         ibis_channels = {
#             "ch0": infant_data.get('ch_0', {}).get('data', []),
#             "ch1": infant_data.get('ch_1', {}).get('data', []),
#             "ch2": infant_data.get('ch_2', {}).get('data', [])
#         }

#         best_ch, results = select_best_channel(ibis_channels, short_channel_pct=short_channel_pct)
#         row = {"subject_id": subj_id, "best_channel": best_ch}

#         if results is None:
#             for ch in ibis_channels:
#                 for m in expected_metrics:
#                     row[f"{m}_{ch}"] = np.nan
#                     row[f"{m}_rank_{ch}"] = np.nan
#                 row[f"total_score_{ch}"] = np.nan
#             row["best_channel_IBIs"] = None
#             row["best_total_score"] = np.nan
#             rows.append(row)
#             continue

#         # Fill metrics and ranks
#         for ch in ibis_channels:
#             metrics = results.get("metrics", {}).get(ch)
#             ranks = results.get("ranks", {}).get(ch)
#             if metrics is None:
#                 for m in expected_metrics:
#                     row[f"{m}_{ch}"] = np.nan
#                     row[f"{m}_rank_{ch}"] = np.nan
#                 row[f"total_score_{ch}"] = np.nan
#             else:
#                 for m, value in metrics.items():
#                     row[f"{m}_{ch}"] = value
#                     row[f"{m}_rank_{ch}"] = ranks.get(m, np.nan)
#                 row[f"total_score_{ch}"] = results.get("total_scores", {}).get(ch, np.nan)

#         # Best channel data
#         if best_ch is not None and best_ch in ibis_channels:
#             best_ibis = ibis_channels[best_ch]
#             if isinstance(best_ibis, np.ndarray):
#                 best_ibis = best_ibis.tolist()
#             row["best_channel_IBIs"] = best_ibis

#             best_metrics = results.get("metrics", {}).get(best_ch, {})
#             for m in expected_metrics:
#                 row[f"best_{m}"] = best_metrics.get(m, np.nan)
#                 row[f"best_{m}_rank"] = results.get("ranks", {}).get(best_ch, {}).get(m, np.nan)
#             row["best_total_score"] = results.get("total_scores", {}).get(best_ch, np.nan)
#         else:
#             row["best_channel_IBIs"] = None
#             for m in expected_metrics:
#                 row[f"best_{m}"] = np.nan
#                 row[f"best_{m}_rank"] = np.nan
#             row["best_total_score"] = np.nan

#         rows.append(row)

#     return pd.DataFrame(rows)




import numpy as np
import pandas as pd

# -----------------------------
# HRV Metric Computation
# -----------------------------
def compute_metrics(ibis_ms, long_ibi_threshold=1000):
    """
    Compute key IBI and HRV metrics for one channel.

    Parameters
    ----------
    ibis_ms : list or array of floats
        Inter-beat intervals (ms).
    long_ibi_threshold : float, optional
        Threshold for detecting unusually long IBIs (default = 1000 ms).

    Returns
    -------
    dict
        - sdrr: Standard deviation of IBIs (overall variability)
        - long_ibi_threshold: Number of IBIs > threshold (potential pauses)
        - rmssd: Root Mean Square of Successive Differences (short-term HRV)
        - pnn50: % of successive IBIs differing by >50 ms
          → Both RMSSD and pNN50 reflect parasympathetic activity
            (Shaffer & Ginsberg, 2017)
    """
    ibis = np.array(ibis_ms)
    ibis = ibis[~np.isnan(ibis)]
    if len(ibis) < 2:
        return {"sdrr": np.nan, "long_ibi_threshold": np.nan,
                "rmssd": np.nan, "pnn50": np.nan}

    diffs = np.diff(ibis)
    return {
        "sdrr": np.std(ibis, ddof=1),
        "long_ibi_threshold": np.sum(ibis > long_ibi_threshold),
        "rmssd": np.sqrt(np.mean(diffs**2)),
        "pnn50": np.sum(np.abs(diffs) > 50) / len(diffs) * 100
    }

# -----------------------------
# Channel Agreement (new)
# -----------------------------
def compute_channel_consistency(ibis_channels):
    """
    Compute pairwise channel agreement based on mean absolute differences.
    Lower = better agreement.

    Returns
    -------
    pairwise_diffs : dict of pairwise mean abs differences
    channel_consistency : dict of per-channel average difference (lower = cleaner)
    """
    ch_names = list(ibis_channels.keys())
    max_len = max(len(ibis_channels[ch]) for ch in ch_names)

    # Pad all channels with NaNs (cast to float)
    padded = {
        ch: np.pad(
            np.asarray(ibis_channels[ch], dtype=float),  # cast to float
            (0, max_len - len(ibis_channels[ch])),
            constant_values=np.nan
        )
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

    channel_consistency = {
        ch: np.nanmean(diffs) if diffs else np.nan
        for ch, diffs in ch_diffs.items()
    }

    return pairwise_diffs, channel_consistency

# -----------------------------
#  Channel Selection Logic
# -----------------------------
def select_best_channel(ibis_channels, short_channel_pct=0.90):
    """
    Select the best ECG channel based on:
    - IBI/HRV metrics
    - Inter-channel consistency

    Returns
    -------
    best_ch : str
        Channel name with the lowest (best) total score.
    results : dict
        Detailed metrics, ranks, and consistency measures.
    """
    metrics_per_ch = {}
    n_beats = {ch: len(ibis) for ch, ibis in ibis_channels.items()}
    max_len = max(n_beats.values())

    # Filter out short channels
    valid_channels = [ch for ch, n in n_beats.items() if n >= short_channel_pct * max_len]
    if not valid_channels:
        return None, None

    # HRV metrics
    for ch in valid_channels:
        metrics_per_ch[ch] = compute_metrics(ibis_channels[ch])

    # Consistency metrics
    _, channel_consistency = compute_channel_consistency(ibis_channels)

    # Ranking and total scores
    total_scores = {}
    for ch in valid_channels:
        m = metrics_per_ch[ch]
        # Higher SDRR/long_ibi → more unstable
        score = (
            (m["sdrr"] or 0) * 1.0 +
            (m["long_ibi_threshold"] or 0) * 2.0 -
            (m["rmssd"] or 0) * 0.5 -
            (m["pnn50"] or 0) * 0.5 +
            (channel_consistency.get(ch, np.nan) or 0) * 1
        )
        total_scores[ch] = score

    best_ch = min(total_scores, key=total_scores.get)

    return best_ch, {
        "metrics": metrics_per_ch,
        "consistency": channel_consistency,
        "total_scores": total_scores
    }

# -----------------------------
# 4DataFrame Builder
# -----------------------------
def build_best_channel_df(ibis_data_dict, short_channel_pct=0.90):
    """
    Builds a summary DataFrame across all subjects.
    """
    rows = []
    expected_metrics = ["sdrr", "long_ibi_threshold", "rmssd", "pnn50"]

    for subj_id, subj_data in ibis_data_dict.items():
        infant_data = subj_data.get("infant", {})
        ibis_channels = {
            "ch0": infant_data.get("ch_0", {}).get("data", []),
            "ch1": infant_data.get("ch_1", {}).get("data", []),
            "ch2": infant_data.get("ch_2", {}).get("data", [])
        }

        best_ch, results = select_best_channel(ibis_channels, short_channel_pct)
        row = {"subject_id": subj_id, "best_channel": best_ch}

        if results is None:
            rows.append(row)
            continue

        # Add per-channel metrics and consistency
        for ch in ibis_channels:
            metrics = results["metrics"].get(ch, {})
            for m in expected_metrics:
                row[f"{m}_{ch}"] = metrics.get(m, np.nan)
            row[f"consistency_{ch}"] = results["consistency"].get(ch, np.nan)
            row[f"total_score_{ch}"] = results["total_scores"].get(ch, np.nan)

        # Add best channel summary
        best_metrics = results["metrics"].get(best_ch, {})
        for m in expected_metrics:
            row[f"best_{m}"] = best_metrics.get(m, np.nan)
        row["best_consistency"] = results["consistency"].get(best_ch, np.nan)
        row["best_total_score"] = results["total_scores"].get(best_ch, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)
