from metrics import compute_metrics
from ranking import rank_channels
from typing import Literal
import numpy as np
import pandas as pd
# from validation import compute_channel_consistency


# -----------------------------
# Build summary DataFrame
# -----------------------------
def channel_selection(ibis_data_dict, participant: Literal['mom', 'infant'], 
                          short_channel_pct= 0.85, weights=None, infant_ibis_th =600, mom_ibis_th = 1000):
    """
    Build a summary DataFrame with channels ordered by rank (best → medium → worst),
    and columns ordered by parameter type: length → median → sdrr → long_ibi_count → mean.
    """
    data_dic = {}


    for subj_id, subj_data in ibis_data_dict.items():
        sub_data = subj_data.get(participant, {})

        # Extract channels
        ibis_channels = {}
        for ch_key in sub_data.keys():
            if 'ch' in ch_key and 'data' in sub_data[ch_key]:
                ibis_channels[ch_key] = sub_data[ch_key]['data']

        # Select best channel + ranks
        best_ch, results = select_best_channel(ibis_channels, participant, short_channel_pct, weights, infant_ibis_th =600, mom_ibis_th = 1000)
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

    return df, data_dic


def select_best_channel(ibis_channels, participant: Literal['mom', 'infant'],
                        short_channel_pct=0.9, weights=None, infant_ibis_th =600, mom_ibis_th = 1000):
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
        metrics = compute_metrics(ibis_channels[ch], participant, infant_ibis_th =600, mom_ibis_th = 1000)
        metrics['length'] = n_ibis[ch]
        metrics['invalid'] = not valid_flags[ch]
        metrics_per_ch[ch] = metrics


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
