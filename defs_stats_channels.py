import pandas as pd

def channel_stats_df(data_dict, participant, group, condition, with_all_indices= False):
    """
    Returns a DataFrame with dyad_id as index and columns like:
    ch_0_mean, ch_0_std, ch_0_min, ch_0_max, ch_0_count, ch_1_mean, ...
    """
    all_dyads = {}

    for dyad_id, dyad_data in data_dict[group].items():
        if condition not in dyad_data or participant not in dyad_data[condition]:
            continue

        participant_channels = dyad_data[condition][participant]
        dyad_stats = {}

        for channel_name, channel_dict in participant_channels.items():
            series = channel_dict['data']
            # ensure it's always treated as a pandas Series
            if not isinstance(series, pd.Series):
                series = pd.Series([series])

            dyad_stats[f"{channel_name}_mean"] = series.mean()
            dyad_stats[f"{channel_name}_std"] = series.std()
            dyad_stats[f"{channel_name}_min"] = series.min()
            dyad_stats[f"{channel_name}_max"] = series.max()
            dyad_stats[f"{channel_name}_lenght"] = len(series)
            dyad_stats[f"{channel_name}_above_1000"] = (series > 1000).sum()  # new counter

        all_dyads[dyad_id] = dyad_stats

    df = pd.DataFrame.from_dict(all_dyads, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()

    df.index.name = f'dyad_id ({len(all_dyads)} dyads)'

    cols = df.columns
    order = (
        sorted([c for c in cols if c.endswith("lenght")]) +
        sorted([c for c in cols if c.endswith("mean")]) +
        sorted([c for c in cols if c.endswith("std")]) +
        sorted([c for c in cols if c.endswith("min")]) +
        sorted([c for c in cols if c.endswith("max")]) +
        sorted([c for c in cols if c.endswith("above_1000")])
    )
    df = df[order]
    
    if with_all_indices == True:
        df = df.reindex(range(1, 91))

    return df

def channel_rsa_stats_df(data_dict, participant, group, condition, with_all_indices=False):
    """
    Returns a DataFrame with dyad_id as index and columns like:
    Present channels, ch_0_mean, ch_0_std, ch_0_min, ch_0_max, ch_0_length, ch_1_mean, ...
    
    participant: 'motherRsa' or 'infantRsa'
    """
    all_dyads = {}

    for dyad_id, dyad_data in data_dict[group].items():
        if condition not in dyad_data:
            continue

        dyad_stats = {}
        present_channels = []

        for channel_name, channel_dict in dyad_data[condition].items():
            df = channel_dict['data']
            series = df[participant]

            if series.notna().any():
                present_channels.append(channel_name)

            dyad_stats[f"{channel_name}_mean"] = series.mean()
            dyad_stats[f"{channel_name}_std"] = series.std()
            dyad_stats[f"{channel_name}_min"] = series.min()
            dyad_stats[f"{channel_name}_max"] = series.max()
            dyad_stats[f"{channel_name}_length"] = len(series)

        # Add Present channels as first column
        dyad_stats['Present channels'] = ', '.join(present_channels)
        all_dyads[dyad_id] = dyad_stats

    df = pd.DataFrame.from_dict(all_dyads, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()
    df.index.name = f'dyad_id ({len(all_dyads)} dyads)'

    # Move "Present channels" to first column
    if 'Present channels' in df.columns:
        cols = ['Present channels'] + [c for c in df.columns if c != 'Present channels']
        df = df[cols]

    # Reorder other stats columns
    cols = df.columns
    order = (
        ['Present channels'] +
        sorted([c for c in cols if c.endswith("length")]) +
        sorted([c for c in cols if c.endswith("mean")]) +
        sorted([c for c in cols if c.endswith("std")]) +
        sorted([c for c in cols if c.endswith("min")]) +
        sorted([c for c in cols if c.endswith("max")])
    )
    df = df[order]

    if with_all_indices:
        df = df.reindex(range(1, 91))

    return df


def num_over_1000(datasets: dict):
    """
    datasets: dict of {name: DataFrame} where DataFrame comes from channel_stats_df
    Returns one summary DataFrame:
        rows = dataset names
        cols = channels (ch_0, ch_1, ...)
    """
    summary_dict = {}
    dyads_total = {}
    dyads_above_1000 = {}

    for name, df in datasets.items():
        above_1000 = df.filter(like="_above_1000")
        counts = (above_1000 > 0).sum()
        counts.index = counts.index.str.replace("_above_1000", "", regex=False)
        summary_dict[name] = counts
        dyads_total[name] = int((df.index.name).split("(")[1].split()[0])
        dyads_above_1000[name] = (above_1000.gt(0).any(axis=1)).sum()

    summary_df = pd.DataFrame(summary_dict).T.fillna(0).astype(int)
    summary_df["dyads_total"] = pd.Series(dyads_total)
    summary_df["dyads_above_1000"] = pd.Series(dyads_above_1000)

    return summary_df
