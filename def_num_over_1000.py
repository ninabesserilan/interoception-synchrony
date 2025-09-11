import pandas as pd

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
