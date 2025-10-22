import numpy as np
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
