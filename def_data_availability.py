import pandas as pd

def data_availability_df(data_dict):
    """
    Returns a DataFrame with dyad_id as index.
    Columns: condition_participant_channel (e.g., with_toys_infant_ch_0).
    Values: True/False depending on whether data exists.
    """

    availability = {}

    for group_name, group_data in data_dict.items():
        # Map group to suffix (9 or 18)
        group_suffix = "9" if "9_months" in group_name else "18"

        for dyad_id, dyad_data in group_data.items():
            if dyad_id not in availability:
                availability[dyad_id] = {col: "False" for col in [
                    f"infant_9_ch_0", f"infant_9_ch_1", f"infant_9_ch_2",
                    f"mom_9_ch_0", f"mom_9_ch_1", f"mom_9_ch_2",
                    f"infant_18_ch_0", f"infant_18_ch_1", f"infant_18_ch_2",
                    f"mom_18_ch_0", f"mom_18_ch_1", f"mom_18_ch_2"
                ]}

            for condition, cond_data in dyad_data.items():
                for participant, part_data in cond_data.items():
                    for channel_name in part_data.keys():
                        col_name = f"{participant}_{group_suffix}_{channel_name}"
                        current_val = availability[dyad_id][col_name]

                        if current_val == "False":
                            availability[dyad_id][col_name] = [condition]
                        else:
                            availability[dyad_id][col_name].append(condition)

    df = pd.DataFrame.from_dict(availability, orient="index")
    df.index = df.index.astype(int)
    df = df.sort_index()

    df.index.name = f"dyad_id ({df.shape[0]} dyads)"
    return df


def availability_view_by_participant(df, with_all_indices= False):
    """
    Transform the output of data_availability_df into a compact 4-column view:
    "infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month".

    Each cell contains a string:
        condition: channel_numbers_comma_separated
        or "False" if no data exists
    """
    out_columns = ["infant 9 month", "infant 18 month", "mom 18 month", "mom 9 month"]

    # Initialize a dict to store data per dyad
    dyad_data_dict = {dyad: {col: {} for col in out_columns} for dyad in df.index}

    # Step 1: Collect channels per condition
    for dyad in df.index:
        for col in df.columns:
            val = df.loc[dyad, col]
            if val == "False":
                continue

            participant = col.split("_")[0]      # 'infant' or 'mom'
            group = col.split("_")[1]           # '9' or '18'
            ch_num = int(col.split("_")[-1])    # last part -> 0,1,2
            age_col = f"{participant} {group} month"

            for condition in val:
                dyad_data_dict[dyad][age_col].setdefault(condition, set()).add(ch_num)

    # Step 2: Build the final string per cell
    for dyad in dyad_data_dict:
        for age_col in out_columns:
            cond_dict = dyad_data_dict[dyad][age_col]
            if cond_dict:
                lines = []
                for condition, channels in cond_dict.items():
                    channels_str = ",".join(map(str, sorted(channels)))
                    lines.append(f"{condition}: {channels_str}")
                dyad_data_dict[dyad][age_col] = "  ".join(lines)
                dyad_data_dict[dyad][age_col] = str(dyad_data_dict[dyad][age_col])
            else:
                dyad_data_dict[dyad][age_col] = "False"

    # Step 3: Create DataFrame
    out_df = pd.DataFrame.from_dict(dyad_data_dict, orient="index")
    out_df.index.name = df.index.name
    out_df.index = out_df.index.astype(int)
    out_df = out_df.sort_index()

    if with_all_indices == True:
        out_df = out_df.reindex(range(1, 91))

    return out_df


def interleaved_availability_diff(ibis_df, peaks_df, months_first="18"):
    """
    Generate an interleaved comparison of ibis vs peaks availability DataFrames.

    Parameters:
    - ibis_df: DataFrame from availability_view_by_participant for ibis
    - peaks_df: DataFrame from availability_view_by_participant for peaks
    - months_first: "9" or "18" — which months appear first in the columns

    Returns:
    - diff_comparison: DataFrame with dyads where ibis and peaks differ,
      columns interleaved: ibis_X, peaks_X
    """
    # Compare the two views
    is_same = peaks_df == ibis_df

    # Find dyads where any column differs
    dyads_diff = is_same.index[~is_same.all(axis=1)]

    if len(dyads_diff) == 0:
        return pd.DataFrame()  # No differences

    # Subset only differing dyads
    ibis_diff = ibis_df.loc[dyads_diff].add_prefix("ibis_")
    peaks_diff = peaks_df.loc[dyads_diff].add_prefix("peaks_")

    # Define column order
    if months_first == "18":
        categories = ["infant 18 month", "mom 18 month", "infant 9 month", "mom 9 month"]
    else:
        categories = ["infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month"]

    ordered_cols = []
    for cat in categories:
        ordered_cols.append(f"ibis_{cat}")
        ordered_cols.append(f"peaks_{cat}")

    # Concatenate side by side in desired order
    diff_comparison = pd.concat([ibis_diff, peaks_diff], axis=1)[ordered_cols]
    diff_comparison.index = diff_comparison.index.astype(int)
    diff_comparison = diff_comparison.sort_index()

    return diff_comparison
