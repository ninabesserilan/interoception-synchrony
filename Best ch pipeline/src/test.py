from data_loader import ibis_toys_9m_infants_data, ibis_toys_9m_moms_data, ibis_no_toys_9m_infants_data, ibis_no_toys_9m_moms_data # ibis data- 9 month
from metrics import compute_metrics
import numpy as np
from ranking import weights_calculation
### Testing channel_selection

# Data set
sub_id = '49'                           # change as you want
data_dict = ibis_toys_9m_moms_data   # change as you want
participant = 'mom'                  # change as you want

# Data
sub_data = (data_dict[sub_id]).get(participant, {})

# Variables
short_channel_pct = 0.80
infant_ibis_th = 600
mom_ibis_th = 1000

# Extract channels
ibis_channels = {}
for ch_key in sub_data.keys():
    if 'ch' in ch_key and 'data' in sub_data[ch_key]:
        ibis_channels[ch_key] = sub_data[ch_key]['data']


# testing select_best_channel
n_ibis = {ch: len(np.atleast_1d(ibis)) for ch, ibis in ibis_channels.items()}
max_len = max(n_ibis.values()) if n_ibis else 0
valid_flags = {ch: (n_ibis[ch] >= short_channel_pct * max_len) for ch in n_ibis}

metrics_per_ch = {}
for ch in ibis_channels:
    metrics = compute_metrics(ibis_channels[ch], participant, infant_ibis_th, mom_ibis_th)
    metrics['length'] = n_ibis[ch]
    metrics['invalid'] = not valid_flags[ch]
    metrics_per_ch[ch] = metrics


invalid_channels = [ch for ch, m in metrics_per_ch.items() if m.get('invalid', False)]
valid_channels = [ch for ch in metrics_per_ch.keys() if ch not in invalid_channels]


ch_auto_rank = None # Channel that are automatically ranked, without the ranking pipeline

channels_to_rank = metrics_per_ch.keys()

if len(invalid_channels) == 1:
    # One invalid → it's automatically worst
    ch_auto_rank = invalid_channels[0]
    worst_ch = ch_auto_rank
    channels_to_rank = {ch: metrics_per_ch[ch] for ch in valid_channels if ch in metrics_per_ch}

elif len(invalid_channels) == 2:
    # Two invalid → valid is automatically best
    ch_auto_rank = valid_channels[0]
    best_ch = ch_auto_rank
    channels_to_rank = {ch: metrics_per_ch[ch] for ch in invalid_channels if ch in metrics_per_ch}


directions = {
    'sdrr': False,           # lower is better
    'long_ibi_count': False, # lower is better
}

ranks = {ch: {} for ch in channels_to_rank.keys()}

for metric, higher_is_better in directions.items():
    # Sort channels by metric
    ch_vals = [(ch, channels_to_rank[ch][metric]) for ch in channels_to_rank if not np.isnan(channels_to_rank[ch][metric])]
    sorted_chs = sorted(ch_vals, key=lambda x: x[1], reverse=higher_is_better)

    # Tie-aware ranking
    rank_val = 1
    last_val = None
    for idx, (ch, val) in enumerate(sorted_chs):
        if last_val is not None and val != last_val:
            rank_val = idx + 1
        ranks[ch][metric + '_rank'] = rank_val 
        last_val = val

    # Assign worst rank to channels missing metric or nan
    ranked_chs = {ch for ch, _ in sorted_chs}
    worst_rank = len(sorted_chs) + 1
    for ch in channels_to_rank.keys():
        if ch not in ranked_chs:
            ranks[ch][metric + '_rank'] = worst_rank 

if ch_auto_rank != None:
    if best_ch != None:
        new_ranks = {'sdrr_rank': 1, 'long_ibi_count_rank': 1}
        for ch, metrics in ranks.items():
            for key in metrics:
                metrics[key] += 1
    elif worst_ch != None:
        new_ranks = {'sdrr_rank': 3, 'long_ibi_count_rank': 3}


weights_for_rank = weights_calculation(channels_to_rank, weights=None)
ranks[ch_auto_rank] = new_ranks
ranks
# Apply weights
weighted_ranks = {
    ch: {k: v * weights_for_rank.get(k, 1) for k, v in metrics.items()}
    for ch, metrics in ranks.items()
}

# Apply weights
weighted_ranks = {
    ch: {k: v * weights_for_rank.get(k, 1) for k, v in metrics.items()}
    for ch, metrics in ranks.items()
}


sum_ranks = {ch: sum(v.values()) for ch, v in weighted_ranks.items()}
# sort channels by sum_rank (ascending = better)
sorted_chs = sorted(sum_ranks.keys(), key=lambda ch: sum_ranks[ch])
sorted_chs
# tie-aware sorting within each rank
final_order = []
i = 0
while i < len(sorted_chs):
    ch = sorted_chs[i]
    rank_val = sum_ranks[ch]
    
    # find all channels with the same rank
    tied = [ch]
    j = i + 1
    while j < len(sorted_chs) and sum_ranks[sorted_chs[j]] == rank_val:
        tied.append(sorted_chs[j])
        j += 1
    
    # break tie by length
    tied_sorted = sorted(tied, key=lambda ch: metrics_per_ch[ch].get('length', 0), reverse=True)
    final_order.extend(tied_sorted)
    
    i = j
    # Assign ordinal ranks: 1, 2, 3, ...
best_ch = final_order[0]
total_ranks = {ch: i + 1 for i, ch in enumerate(final_order)}

# for ch, metrics in ranks.items():
#     ranks[ch][metric + '_rank'] = rank_val * weights_for_rank.get(metric, 1.0)


# weighted_ranks
# # Compute sum rank per channel
# sum_ranks = {ch: sum(v.values()) for ch, v in ranks.items()}

# # sort channels by sum_rank (ascending = better)
# sorted_chs = sorted(sum_ranks.keys(), key=lambda ch: sum_ranks[ch])

# # tie-aware sorting within each rank
# final_order = []
# i = 0
# while i < len(sorted_chs):
#     ch = sorted_chs[i]
#     rank_val = sum_ranks[ch]
    
#     # find all channels with the same rank
#     tied = [ch]
#     j = i + 1
#     while j < len(sorted_chs) and sum_ranks[sorted_chs[j]] == rank_val:
#         tied.append(sorted_chs[j])
#         j += 1
    
#     # break tie by length
#     tied_sorted = sorted(tied, key=lambda ch: channels_to_rank[ch].get('length', 0), reverse=True)
#     final_order.extend(tied_sorted)
    
#     i = j
#     # Assign ordinal ranks: 1, 2, 3, ...
# best_ch = final_order[0]
# total_ranks = {ch: i + 1 for i, ch in enumerate(final_order)}






# * weights.get(metric, 1.0)


# channels_to_keep = ['ch_0', 'ch_1']
# new_metrics = {ch: metrics_per_ch[ch] for ch in channels_to_keep if ch in metrics_per_ch}


# directions = {
#     'sdrr': False,           # lower is better
#     'long_ibi_count': False, # lower is better
# }

# ranks = {ch: {} for ch in metrics_per_ch.keys()}

# for metric, higher_is_better in directions.items():
#     # Sort channels by metric
#     ch_vals = [(ch, metrics_per_ch[ch][metric]) for ch in metrics_per_ch if not np.isnan(metrics_per_ch[ch][metric])]
#     sorted_chs = sorted(ch_vals, key=lambda x: x[1], reverse=higher_is_better)

#     # Tie-aware ranking
#     rank_val = 1
#     last_val = None
#     for idx, (ch, val) in enumerate(sorted_chs):
#         if last_val is not None and val != last_val:
#             rank_val = idx + 1
#         ranks[ch][metric + '_rank'] = rank_val * weights.get(metric, 1.0)
#         last_val = val

#     # Assign worst rank to channels missing metric or nan
#     ranked_chs = {ch for ch, _ in sorted_chs}
#     worst_rank = len(sorted_chs) + 1
#     for ch in metrics_per_ch.keys():
#         if ch not in ranked_chs:
#             ranks[ch][metric + '_rank'] = worst_rank * weights.get(metric, 1.0)

# # Compute sum rank per channel
# sum_ranks = {ch: sum(v.values()) for ch, v in ranks.items()}

# # sort channels by sum_rank (ascending = better)
# sorted_chs = sorted(sum_ranks.keys(), key=lambda ch: sum_ranks[ch])

# # tie-aware sorting within each rank
# final_order = []
# i = 0
# while i < len(sorted_chs):
#     ch = sorted_chs[i]
#     rank_val = sum_ranks[ch]
    
#     # find all channels with the same rank
#     tied = [ch]
#     j = i + 1
#     while j < len(sorted_chs) and sum_ranks[sorted_chs[j]] == rank_val:
#         tied.append(sorted_chs[j])
#         j += 1
    
#     # break tie by length
#     tied_sorted = sorted(tied, key=lambda ch: metrics_per_ch[ch].get('length', 0), reverse=True)
#     final_order.extend(tied_sorted)
    
#     i = j
#     # Assign ordinal ranks: 1, 2, 3, ...
# best_ch = final_order[0]