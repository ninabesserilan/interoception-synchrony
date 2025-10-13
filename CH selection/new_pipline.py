
import numpy as np
import pandas as pd


def compute_metrics(ibis_ms, participant):
    """
    Compute key IBI and HRV metrics for one channel.
    Returns dictionary with metrics.

        - sdrr: Standard deviation of IBIs (overall variability)
        - long_ibi_threshold: Number of IBIs > threshold (potential pauses)
        - rmssd: Root Mean Square of Successive Differences (short-term HRV)
        - pnn50: % of successive IBIs differing by >50 ms
          → Both RMSSD and pNN50 reflect parasympathetic activity
            (Shaffer & Ginsberg, 2017)


    """
    ibis = np.atleast_1d(ibis_ms).astype(float)
    ibis = ibis[~np.isnan(ibis)]
    if len(ibis) < 2:
        return {"sdrr": np.nan, "long_ibi_count": np.nan,
                "rmssd": np.nan, "pnn50": np.nan}

    # Thresholds per participant type
    if participant == 'mom':
        long_ibi_threshold = 600
    elif participant == 'infant':
        long_ibi_threshold = 1000
    else:
        long_ibi_threshold = 800

    diffs = np.diff(ibis)
    return {
        "sdrr": np.std(ibis, ddof=1),
        "long_ibi_count": np.sum(ibis > long_ibi_threshold),
        "rmssd": np.sqrt(np.mean(diffs ** 2)),
        "pnn50": np.sum(np.abs(diffs) > 50) / len(diffs) * 100
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
# Rank channels using dicts
# -----------------------------
def rank_channels(metrics_per_ch, weights):
    """
    Rank channels using weights in a dict-based approach.
    Lower total score = better channel.
    """
    directions = {
        'sdrr': False,           # lower is better
        'long_ibi_count': False, # lower is better
        'rmssd': False,          # lower is better
        'pnn50': True            # higher is better
    }

    ranks = {ch: {} for ch in metrics_per_ch.keys()}

    for metric, higher_is_better in directions.items():
        # Sort channels by metric
        sorted_chs = sorted(
            metrics_per_ch.items(),
            key=lambda x: x[1][metric] if x[1][metric] is not None else np.inf,
            reverse=higher_is_better
        )
        # Assign ranks with weight
        for rank, (ch, _) in enumerate(sorted_chs, start=1):
            ranks[ch][metric + '_rank'] = rank * weights.get(metric, 1.0)

    # Compute total rank per channel
    total_ranks = {ch: sum(v.values()) for ch, v in ranks.items()}
    best_ch = min(total_ranks, key=total_ranks.get)
    return best_ch, ranks, total_ranks

# -----------------------------
# Select best channel
# -----------------------------
def select_best_channel(ibis_channels, participant, short_channel_pct=0.9, weights=None):
    if weights is None:
        weights = {'sdrr': 1.0, 'long_ibi_count': 2.0, 'rmssd': 0.5, 'pnn50': -0.5}

    # Filter channels by length
    n_ibis = {ch: len(np.atleast_1d(ibis)) for ch, ibis in ibis_channels.items()}
    max_len = max(n_ibis.values()) if n_ibis else 0
    valid_channels = [ch for ch, n in n_ibis.items() if n >= short_channel_pct * max_len]

    if not valid_channels:
        return None, None

    # Compute metrics
    metrics_per_ch = {ch: compute_metrics(ibis_channels[ch], participant)
                      for ch in valid_channels}

    # Compute consistency (optional)
    _, channel_consistency = compute_channel_consistency(ibis_channels)

    # Rank channels
    best_ch, ranks, total_ranks = rank_channels(metrics_per_ch, weights)

    return best_ch, {
        "metrics": metrics_per_ch,
        "consistency": channel_consistency,
        "ranks": ranks,
        "total_ranks": total_ranks
    }

# -----------------------------
# Build summary DataFrame (optional)
# -----------------------------
def build_best_channel_df(ibis_data_dict, participant, short_channel_pct=0.9, weights=None):
    """
    Build a summary DataFrame with best channel per subject.
    Dynamically handles missing channels.
    """
    rows = []
    expected_metrics = ["sdrr", "long_ibi_count", "rmssd", "pnn50"]

    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})

        # Build channel dict from available channels
        ibis_channels = {}
        for ch_key in sub_data.keys():
            if 'ch' in ch_key and 'data' in sub_data[ch_key]:
                ibis_channels[ch_key.replace('_', '')] = sub_data[ch_key]['data']

        # Select best channel
        best_ch, results = select_best_channel(ibis_channels, participant, short_channel_pct, weights)
        row = {"subject_id": subj_id, "best_channel": best_ch}

        if results is None:
            rows.append(row)
            continue

        # Add metrics and ranks only for channels that exist
        for ch in results['metrics'].keys():
            metrics = results["metrics"].get(ch, {})
            for m in expected_metrics:
                row[f"{m}_{ch}"] = metrics.get(m, np.nan)

            # Add ranks and total rank if available
            if 'ranks' in results and ch in results['ranks']:
                row[f"total_rank_{ch}"] = results['total_ranks'].get(ch, np.nan)
                for metric in expected_metrics:
                    row[f"{metric}_rank_{ch}"] = results['ranks'][ch].get(metric + '_rank', np.nan)

        rows.append(row)

    return pd.DataFrame(rows)
