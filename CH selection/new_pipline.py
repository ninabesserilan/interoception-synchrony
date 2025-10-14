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
def select_best_channel(ibis_channels, participant: Literal['mom', 'infant'], short_channel_pct=0.9, weights=None):
    if weights is None:
        weights = {'sdrr': 1.0, 'long_ibi_count': 2.0}

    # Filter channels by length
    n_ibis = {ch: len(np.atleast_1d(ibis)) for ch, ibis in ibis_channels.items()}
    max_len = max(n_ibis.values()) if n_ibis else 0
    valid_channels = [ch for ch, n in n_ibis.items() if n >= short_channel_pct * max_len]

    if not valid_channels:
        return None, None

    # Compute metrics and add length to metrics for tie-break
    metrics_per_ch = {}
    for ch in valid_channels:
        metrics = compute_metrics(ibis_channels[ch], participant)
        metrics['length'] = n_ibis[ch]
        metrics_per_ch[ch] = metrics

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


def build_best_channel_df(ibis_data_dict, participant: Literal['mom', 'infant'], short_channel_pct=0.9, weights=None):
    """
    Build a comprehensive summary DataFrame containing all channels for each subject,
    including invalid channels, with their metrics, length, mean, and median.
    Additionally, label best, medium, worst channels based on ranking of valid channels per subject.
    """
    rows = []
    expected_metrics = ["sdrr", "long_ibi_count"]

    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})

        # Extract all channels for the subject
        ibis_channels = {}
        for ch_key in sub_data.keys():
            if 'ch' in ch_key and 'data' in sub_data[ch_key]:
                ibis_channels[ch_key.replace('_', '')] = sub_data[ch_key]['data']

        # Determine valid channels using the short_channel_pct criterion
        n_ibis = {ch: len(np.atleast_1d(v)) for ch, v in ibis_channels.items()}
        max_len = max(n_ibis.values()) if n_ibis else 0
        valid_channels = [ch for ch, n in n_ibis.items() if n >= short_channel_pct * max_len]

        # Compute metrics for all channels (valid + invalid)
        all_metrics = {}
        for ch, ibis_vals in ibis_channels.items():
            metrics = compute_metrics(ibis_vals, participant)
            all_metrics[ch] = metrics

        # Select best channel and rankings only from valid channels
        if valid_channels:
            valid_ibis_channels = {ch: ibis_channels[ch] for ch in valid_channels}
            best_ch, results = select_best_channel(valid_ibis_channels, participant, short_channel_pct, weights)
            ranks = results["ranks"]
            total_ranks = results["total_ranks"]
        else:
            best_ch, ranks, total_ranks = None, {}, {}

        # Get a sorted list of valid channels by total rank
        sorted_valid_channels = []
        if total_ranks:
            sorted_valid_channels = [ch for ch, _ in sorted(total_ranks.items(), key=lambda x: x[1])]

        # Identify best, medium, worst channel labels for subject (only among valid channels)
        label_channels = [None, None, None]
        for i in range(min(3, len(sorted_valid_channels))):
            label_channels[i] = sorted_valid_channels[i]

        # Now create one row per channel (valid or invalid) with all data
        for ch in ibis_channels.keys():
            row = {"subject_id": subj_id, "channel": ch}

            ibis_vals = np.atleast_1d(ibis_channels[ch]).astype(float)
            row["length"] = len(ibis_vals)
            row["mean"] = np.nanmean(ibis_vals)
            row["median"] = np.nanmedian(ibis_vals)

            metrics = all_metrics.get(ch, {})
            for m in expected_metrics:
                row[m] = metrics.get(m, np.nan)

            # Label channels if they are best/medium/worst among valid channels
            if ch == label_channels[0]:
                row["label"] = "best_channel"
            elif ch == label_channels[1]:
                row["label"] = "medium_channel"
            elif ch == label_channels[2]:
                row["label"] = "worst_channel"
            else:
                row["label"] = None

            # Add ranking info if available (only valid channels have ranking info)
            if ch in ranks:
                for metric_rank_key, rank_val in ranks[ch].items():
                    row[metric_rank_key] = rank_val
                row["total_rank"] = total_ranks.get(ch, np.nan)
            else:
                row["sdrr_rank"] = np.nan
                row["long_ibi_count_rank"] = np.nan
                row["total_rank"] = np.nan

            rows.append(row)

    # Return a DataFrame with one channel per row including valid/invalid and ranking labels
    return pd.DataFrame(rows)
