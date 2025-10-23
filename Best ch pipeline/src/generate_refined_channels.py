import numpy as np
import pandas as pd
from metrics import compute_metrics
from channel_selection import select_best_channel
from typing import Literal, List, Tuple



def fill_missing_peaks(
    participant: Literal['infant', 'mom'],
    peaks_data_dict,
    ch_selection_dict,
    missing_peaks_dict,
    median_ibis_percantage_th = 0.75
):
    new_best_ch_peaks = {'data': {}, 'source': {}}

    for subj, missing_data in missing_peaks_dict.items():
        #  Access best channel correctly
        best_ch = ch_selection_dict[subj]['best_channel']
        best_peaks = peaks_data_dict[subj][participant][best_ch]['data']
        best_ch_median_ibis= ch_selection_dict[subj]['median_best']

        merge_tolerance_ms = best_ch_median_ibis * median_ibis_percantage_th

        new_peaks = best_peaks.copy()
        new_sources = pd.Series([{ 'best': best_ch }] * len(new_peaks))

        if missing_data:
            for interval_idx, interval_data in missing_data.items():
                # --- Collect candidates from medium + worst ---
                candidates = []
                for quality in ['medium', 'worst']:
                    ch_dict = interval_data['other_channels'].get(quality, {})
                    for ch_name, ch_peaks in ch_dict.items():
                        for p in ch_peaks:
                            candidates.append((p, quality, ch_name))

                # --- Merge close ones ---
                merged_candidates = merge_close_peaks(candidates, merge_tolerance_ms)

                # --- Add merged peaks ---
                for p, quality, ch_name in merged_candidates:
                    if p not in new_peaks.values:
                        new_peaks = pd.concat([new_peaks, pd.Series([p])], ignore_index=True)
                        new_sources = pd.concat(
                            [new_sources, pd.Series([{quality: ch_name}])],
                            ignore_index=True
                        )

        # Sort peaks and reorder sources
        sorted_idx = np.argsort(new_peaks.values)
        new_best_ch_peaks['data'][subj] = new_peaks.iloc[sorted_idx].reset_index(drop=True)
        new_best_ch_peaks['source'][subj] = new_sources.iloc[sorted_idx].reset_index(drop=True)

    return new_best_ch_peaks


def merge_close_peaks(
    peaks_with_meta: List[Tuple[int, str, str]],
    merge_tolerance_ms: float
) -> List[Tuple[int, str, str]]:
    """
    Merge peaks that occur within a given time tolerance, keeping the one
    from the higher-quality channel.

    Parameters
    ----------
    peaks_with_meta : list of tuples
        Each element is (peak_time, quality_label, channel_name)
        Example: [(2626, 'medium', 'ch_0'), (2635, 'worst', 'ch_1')]
    merge_tolerance_ms : int
        Maximum time difference (in ms) between peaks to consider them duplicates.

    Returns
    -------
    merged_peaks : list of tuples
        Same format as input but with merged nearby peaks.
    """

    # Define priority (lower is better)
    QUALITY_PRIORITY = {'best': 0, 'medium': 1, 'worst': 2}

    # Sort by peak time
    peaks_with_meta = sorted(peaks_with_meta, key=lambda x: x[0])

    merged = []
    for p, quality, ch_name in peaks_with_meta:
        if not merged:
            merged.append((p, quality, ch_name))
        else:
            last_p, last_quality, last_ch = merged[-1]
            # If close enough, keep the better one
            if abs(p - last_p) <= merge_tolerance_ms:
                if QUALITY_PRIORITY[quality] < QUALITY_PRIORITY[last_quality]:
                    merged[-1] = (p, quality, ch_name)
            else:
                merged.append((p, quality, ch_name))

    return merged
