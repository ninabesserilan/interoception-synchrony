# import numpy as np

# def compute_channel_consistency(ibis_channels):
#     """
#     Compute average pairwise channel consistency (vectorized).
#     """
#     data = {ch: np.asarray(ibis, dtype=float) for ch, ibis in ibis_channels.items()}
#     max_len = max(len(v) for v in data.values())
#     padded = np.stack([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in data.values()])
#     diffs = np.nanmean(np.abs(padded[:, None, :] - padded[None, :, :]), axis=2)

#     mean_diffs = np.nanmean(diffs, axis=1)
#     return dict(zip(data.keys(), mean_diffs))
