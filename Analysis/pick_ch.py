import numpy as np
import pandas as pd
# -----------------------------
# Metric computation
# -----------------------------
def compute_metrics(ibis_ms, low_ibi_threshold=250, long_ibi_threshold=1000):
    """
    Compute standard and HRV metrics for a single IBI vector (ms)
    """
    ibis = np.array(ibis_ms)
    ibis = ibis[~np.isnan(ibis)]
    if len(ibis) < 2:
        return {
            "std": np.nan,
            "sum_long": np.nan,
            "low_ibi_threshold": 1.0,
            "rmssd": np.nan,
            "nn50": np.nan,
            "pnn50": np.nan
        }

    std_ibi = np.std(ibis, ddof=1)
    sum_long = np.sum(ibis > long_ibi_threshold)
    missing_fraction = np.sum(ibis < low_ibi_threshold)

    # HRV metrics
    diffs = np.diff(ibis)
    rmssd = np.sqrt(np.mean(diffs**2))
    nn50 = np.sum(np.abs(diffs) > 50)
    pnn50 = nn50 / len(diffs) * 100

    return {
        "std": std_ibi,
        "sum_long": sum_long,
        "missing_fraction": missing_fraction,
        "rmssd": rmssd,
        "nn50": nn50,
        "pnn50": pnn50
    }

# -----------------------------
# Ranking function
# -----------------------------
def rank_channels(ch_metrics, metric_name, ascending=True):
    """
    Rank channels for a single metric
    ascending=True → smaller value is better
    """
    values = [ch_metrics[ch][metric_name] for ch in ch_metrics]
    sorted_idx = np.argsort(values if ascending else -np.array(values))
    ranks = {}
    for rank, idx in enumerate(sorted_idx, 1):
        ch = list(ch_metrics.keys())[idx]
        ranks[ch] = rank
    return ranks

# -----------------------------
# Channel selection pipeline
# -----------------------------
def select_best_channel(ibis_channels, short_channel_pct=0.95):
    """
    Select the best ECG channel based on IBI metrics and ranking.
    Median IBI is NOT included anymore.
    """
    metrics_per_ch = {}
    n_beats = {ch: len(ibis) for ch, ibis in ibis_channels.items()}
    max_len = max(n_beats.values())
    
    # Filter out too-short channels
    valid_channels = [ch for ch, length in n_beats.items() if length >= short_channel_pct * max_len]
    if not valid_channels:
        return None, None
    
    # Calculate metrics
    for ch in valid_channels:
        metrics_per_ch[ch] = compute_metrics(ibis_channels[ch])
    
    # Rank channels
    ranks_per_ch = {ch: {} for ch in metrics_per_ch}
    total_scores = {ch: 0 for ch in metrics_per_ch}
    
    for metric in ["std", "sum_long", "missing_fraction", "rmssd", "nn50", "pnn50"]:
        smaller_is_better = metric in ["std", "sum_long", "missing_fraction"]
        values = {ch: metrics_per_ch[ch][metric] for ch in metrics_per_ch}
        sorted_chs = sorted(values, key=lambda ch: values[ch], reverse=not smaller_is_better)
        
        for rank, ch in enumerate(sorted_chs, start=1):
            ranks_per_ch[ch][metric] = rank
            total_scores[ch] += rank
    
    # Best channel = lowest total score
    best_ch = min(total_scores, key=total_scores.get)
    
    results = {
        "metrics": metrics_per_ch,
        "ranks": ranks_per_ch,
        "total_scores": total_scores
    }
    return best_ch, results

# ==========================
# DataFrame builder
# ==========================
# ==========================
def build_best_channel_df(ibis_data_dict, short_channel_pct=0.95):
    """
    data_dict: your dict
    returns: pandas DataFrame with metrics, ranks, scores, and best_channel_IBIs
    """
    rows = []
    expected_metrics = ['std', 'sum_long', 'missing_fraction', 'rmssd', 'nn50', 'pnn50']

    for subj_id, subj_data in ibis_data_dict.items():
        infant_data = subj_data.get('infant', {})
        ibis_channels = {
            "ch0": infant_data.get('ch_0', {}).get('data', []),
            "ch1": infant_data.get('ch_1', {}).get('data', []),
            "ch2": infant_data.get('ch_2', {}).get('data', [])
        }

        best_ch, results = select_best_channel(ibis_channels, short_channel_pct=short_channel_pct)
        row = {"subject_id": subj_id, "best_channel": best_ch}

        if results is None:
            for ch in ibis_channels:
                for m in expected_metrics:
                    row[f"{m}_{ch}"] = np.nan
                    row[f"{m}_rank_{ch}"] = np.nan
                row[f"total_score_{ch}"] = np.nan
            row["best_channel_IBIs"] = None
            row["best_total_score"] = np.nan
            rows.append(row)
            continue

        # Fill metrics and ranks
        for ch in ibis_channels:
            metrics = results.get("metrics", {}).get(ch)
            ranks = results.get("ranks", {}).get(ch)
            if metrics is None:
                for m in expected_metrics:
                    row[f"{m}_{ch}"] = np.nan
                    row[f"{m}_rank_{ch}"] = np.nan
                row[f"total_score_{ch}"] = np.nan
            else:
                for m, value in metrics.items():
                    row[f"{m}_{ch}"] = value
                    row[f"{m}_rank_{ch}"] = ranks.get(m, np.nan)
                row[f"total_score_{ch}"] = results.get("total_scores", {}).get(ch, np.nan)

        # Best channel data
        if best_ch is not None and best_ch in ibis_channels:
            best_ibis = ibis_channels[best_ch]
            if isinstance(best_ibis, np.ndarray):
                best_ibis = best_ibis.tolist()
            row["best_channel_IBIs"] = best_ibis

            best_metrics = results.get("metrics", {}).get(best_ch, {})
            for m in expected_metrics:
                row[f"best_{m}"] = best_metrics.get(m, np.nan)
                row[f"best_{m}_rank"] = results.get("ranks", {}).get(best_ch, {}).get(m, np.nan)
            row["best_total_score"] = results.get("total_scores", {}).get(best_ch, np.nan)
        else:
            row["best_channel_IBIs"] = None
            for m in expected_metrics:
                row[f"best_{m}"] = np.nan
                row[f"best_{m}_rank"] = np.nan
            row["best_total_score"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
