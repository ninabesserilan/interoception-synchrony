
import numpy as np
import pandas as pd


# -----------------------------
#  Channel Selection Logic
# -----------------------------
def select_best_channel(ibis_channels, participant, short_channel_pct=0.90):
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
        metrics_per_ch[ch] = compute_metrics(ibis_channels[ch], participant)

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
# HRV Metric Computation
# -----------------------------
def compute_metrics(ibis_ms, participant):
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

    if participant == 'mom': # NEED TO SWITCH AFTER RECREATING THE PICKLE WITH MOM = ECG1 AND INFANT = ECG2
        long_ibi_threshold=600

    elif participant == 'infant':
        long_ibi_threshold=1000
    
    else:
        long_ibi_threshold=800

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
# Channel Agreement 
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
# DataFrame Builder
# -----------------------------
def build_best_channel_df(ibis_data_dict, participant: str, short_channel_pct=0.90):
    """
    Builds a summary DataFrame across all subjects.
    """
    rows = []
    expected_metrics = ["sdrr", "long_ibi_threshold", "rmssd", "pnn50"]

    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})
        ibis_channels = {
            "ch0": sub_data.get("ch_0", {}).get("data", []),
            "ch1": sub_data.get("ch_1", {}).get("data", []),
            "ch2": sub_data.get("ch_2", {}).get("data", [])
        }

        best_ch, results = select_best_channel(ibis_channels, participant, short_channel_pct)
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
