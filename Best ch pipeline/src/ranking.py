import numpy as np
import pandas as pd

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


