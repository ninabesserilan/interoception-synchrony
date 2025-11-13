from data_loader import ibis_toys_9m_infants_data, ibis_toys_9m_moms_data, ibis_no_toys_9m_infants_data, ibis_no_toys_9m_moms_data # ibis data- 9 month
from data_loader import peaks_toys_9m_infants_data, peaks_toys_9m_moms_data, peaks_no_toys_9m_infants_data, peaks_no_toys_9m_moms_data # peaks data- 9 month

from channel_selection import channel_selection

from metrics import compute_metrics
import numpy as np
from ranking import weights_calculation

df_ch_selection_mom_9m_toys,ch_selection_dict_mom_9m_toys = channel_selection(ibis_toys_9m_moms_data, "mom", short_channel_pct=0.80, infant_ibis_th =600, mom_ibis_th = 1000)
df_ch_selection_infant_9m_toys, ch_selection_dict_infant_9m_toys = channel_selection(ibis_toys_9m_infants_data, "infant", short_channel_pct=0.80, infant_ibis_th =600, mom_ibis_th = 1000)


# Data set
sub_id = '07'                                        # change as you want
ibis_data_dict = ibis_toys_9m_infants_data              # change as you want
peaks_data_dict = peaks_toys_9m_infants_data            # change as you want
participant = 'infant'                                  # change as you want
selection_dict =  ch_selection_dict_infant_9m_toys      # change as you want

# Data
sub_ibis_data = (ibis_data_dict[sub_id]).get(participant, {})
sub_peak_data = (peaks_data_dict[sub_id]).get(participant, {})

short_channel_pct = 0.80
infant_ibis_th = 600
mom_ibis_th = 1000
median_ibis_percantage_th = 0.75

# ### Testing missing peaks
sub_ch_selection_dict = selection_dict[sub_id]   
best_ch = sub_ch_selection_dict['best_channel']
medium_ch = sub_ch_selection_dict['medium_channel']
worst_ch = sub_ch_selection_dict['worst_channel']

missing_peaks_report = {}

best_peaks = np.array(sub_peak_data[best_ch]['data'])
best_ibis = np.diff(best_peaks)

median_ibi = sub_ch_selection_dict['median_best']
threshold = median_ibi * median_ibis_percantage_th

# Prepare dicts for medium & worst
other_channels = {
    'medium': {medium_ch: np.array(sub_peak_data.get(medium_ch, {}).get('data', []))},
    'worst': {worst_ch: np.array(sub_peak_data.get(worst_ch, {}).get('data', []))},
}
refined_best_ch = True
start_time = best_peaks[0]

for ch_type, ch_dict in other_channels.items():
    for ch_name, ch_peaks in ch_dict.items():
        before_start_peaks = ch_peaks[ch_peaks <= start_time - threshold]
        print(f'ch type: {ch_type}, ch name: {ch_name}, before start peaks: {before_start_peaks}')
        if len(before_start_peaks) > 0:
            peaks_to_add = before_start_peaks if refined_best_ch else sorted(
                list(set(np.concatenate([
                    [p, ch_peaks[np.where(ch_peaks == p)[0][0] + 1]]
                    if np.where(ch_peaks == p)[0][0] < len(ch_peaks) - 1 else [p]
                    for p in before_start_peaks
                ])))
            )

            missing_peaks_report.setdefault('before_start', {
                'best_channel_peaks': [int(start_time)],
                'other_channels': {'medium': {}, 'worst': {}}
            })
            missing_peaks_report['before_start']['other_channels'][ch_type][ch_name] = list(map(int, peaks_to_add))

    # --- Peaks after the last peak ---
    end_time = best_peaks[-1]
for ch_type, ch_dict in other_channels.items():
    for ch_name, ch_peaks in ch_dict.items():
        after_end_peaks = ch_peaks[ch_peaks >= end_time + threshold]
        if len(after_end_peaks) > 0:
            peaks_to_add = after_end_peaks if refined_best_ch else sorted( # For debuging, if refined_best_ch = False, the code expands each early peak to include its neighboring peak
                list(set(np.concatenate([
                    [ch_peaks[np.where(ch_peaks == p)[0][0] - 1], p]
                    if np.where(ch_peaks == p)[0][0] > 0 else [p]
                    for p in after_end_peaks
                ])))
            )
            missing_peaks_report.setdefault('after_end', {
                'best_channel_peaks': [int(end_time)],
                'other_channels': {'medium': {}, 'worst': {}}
            })
            missing_peaks_report['after_end']['other_channels'][ch_type][ch_name] = list(map(int, peaks_to_add))

# --- Missing peaks inside intervals ---
for ibi_idx in range(len(best_ibis)):
    interval_start = best_peaks[ibi_idx]
    interval_end = best_peaks[ibi_idx + 1]
    missing_per_type = {'medium': {}, 'worst': {}}

    for ch_type, ch_dict in other_channels.items():
        for ch_name, ch_peaks in ch_dict.items():
            peaks_in_interval = ch_peaks[(ch_peaks > interval_start) & (ch_peaks < interval_end)]
            important_peaks = [
                pt for pt in peaks_in_interval
                if (pt - interval_start) > threshold and (interval_end - pt) > threshold
            ]

            if important_peaks:
                if refined_best_ch:
                    extended_peaks = important_peaks
                else:
                    extended_peaks = sorted(list(set( # For debuging, if refined_best_ch = False, the code expands each early peak to include its neighboring peak
                        np.concatenate([
                            [ch_peaks[np.where(ch_peaks == pt)[0][0] - 1], pt, ch_peaks[np.where(ch_peaks == pt)[0][0] + 1]]
                            if 0 < np.where(ch_peaks == pt)[0][0] < len(ch_peaks) - 1 else [pt]
                            for pt in important_peaks
                        ])
                    )))
                missing_per_type[ch_type][ch_name] = list(map(int, extended_peaks))

    if any(missing_per_type[ch_type] for ch_type in missing_per_type):
        best_peaks_in_interval = [int(p) for p in best_peaks if interval_start <= p <= interval_end]
        missing_peaks_report[ibi_idx] = {
            'best_channel_peaks': best_peaks_in_interval,
            'other_channels': missing_per_type
        }

### Testing channel_selection

# # Extract channels
# ibis_channels = {}
# for ch_key in sub_data.keys():
#     if 'ch' in ch_key and 'data' in sub_data[ch_key]:
#         ibis_channels[ch_key] = sub_data[ch_key]['data']


# # testing select_best_channel
# n_ibis = {ch: len(np.atleast_1d(ibis)) for ch, ibis in ibis_channels.items()}
# max_len = max(n_ibis.values()) if n_ibis else 0
# valid_flags = {ch: (n_ibis[ch] >= short_channel_pct * max_len) for ch in n_ibis}

# metrics_per_ch = {}
# for ch in ibis_channels:
#     metrics = compute_metrics(ibis_channels[ch], participant, infant_ibis_th, mom_ibis_th)
#     metrics['length'] = n_ibis[ch]
#     metrics['invalid'] = not valid_flags[ch]
#     metrics_per_ch[ch] = metrics


# invalid_channels = [ch for ch, m in metrics_per_ch.items() if m.get('invalid', False)]
# valid_channels = [ch for ch in metrics_per_ch.keys() if ch not in invalid_channels]


# ch_auto_rank = None # Channel that are automatically ranked, without the ranking pipeline

# channels_to_rank = metrics_per_ch.keys()

# if len(invalid_channels) == 1:
#     # One invalid → it's automatically worst
#     ch_auto_rank = invalid_channels[0]
#     worst_ch = ch_auto_rank
#     channels_to_rank = {ch: metrics_per_ch[ch] for ch in valid_channels if ch in metrics_per_ch}

# elif len(invalid_channels) == 2:
#     # Two invalid → valid is automatically best
#     ch_auto_rank = valid_channels[0]
#     best_ch = ch_auto_rank
#     channels_to_rank = {ch: metrics_per_ch[ch] for ch in invalid_channels if ch in metrics_per_ch}


# directions = {
#     'sdrr': False,           # lower is better
#     'long_ibi_count': False, # lower is better
# }

# ranks = {ch: {} for ch in channels_to_rank.keys()}

# for metric, higher_is_better in directions.items():
#     # Sort channels by metric
#     ch_vals = [(ch, channels_to_rank[ch][metric]) for ch in channels_to_rank if not np.isnan(channels_to_rank[ch][metric])]
#     sorted_chs = sorted(ch_vals, key=lambda x: x[1], reverse=higher_is_better)

#     # Tie-aware ranking
#     rank_val = 1
#     last_val = None
#     for idx, (ch, val) in enumerate(sorted_chs):
#         if last_val is not None and val != last_val:
#             rank_val = idx + 1
#         ranks[ch][metric + '_rank'] = rank_val 
#         last_val = val

#     # Assign worst rank to channels missing metric or nan
#     ranked_chs = {ch for ch, _ in sorted_chs}
#     worst_rank = len(sorted_chs) + 1
#     for ch in channels_to_rank.keys():
#         if ch not in ranked_chs:
#             ranks[ch][metric + '_rank'] = worst_rank 

# if ch_auto_rank != None:
#     if best_ch != None:
#         new_ranks = {'sdrr_rank': 1, 'long_ibi_count_rank': 1}
#         for ch, metrics in ranks.items():
#             for key in metrics:
#                 metrics[key] += 1
#     elif worst_ch != None:
#         new_ranks = {'sdrr_rank': 3, 'long_ibi_count_rank': 3}


# weights_for_rank = weights_calculation(channels_to_rank, weights=None)
# ranks[ch_auto_rank] = new_ranks
# ranks
# # Apply weights
# weighted_ranks = {
#     ch: {k: v * weights_for_rank.get(k, 1) for k, v in metrics.items()}
#     for ch, metrics in ranks.items()
# }

# # Apply weights
# weighted_ranks = {
#     ch: {k: v * weights_for_rank.get(k, 1) for k, v in metrics.items()}
#     for ch, metrics in ranks.items()
# }


# sum_ranks = {ch: sum(v.values()) for ch, v in weighted_ranks.items()}
# # sort channels by sum_rank (ascending = better)
# sorted_chs = sorted(sum_ranks.keys(), key=lambda ch: sum_ranks[ch])
# sorted_chs
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
# total_ranks = {ch: i + 1 for i, ch in enumerate(final_order)}

