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
    expected_metrics = ["sdrr", "long_ibi_count"]

    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})

        # Extract channels
        ibis_channels = {}
        for ch_key in sub_data.keys():
            if 'ch' in ch_key and 'data' in sub_data[ch_key]:
                ibis_channels[ch_key.replace('_', '')] = sub_data[ch_key]['data']

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

    return df
