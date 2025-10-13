def filter_group_condition(data, group, condition, participant):
    """
    Extract all data for a given group (age) and condition.
    
    Args:
        data (dict): The full ibis_data dict.
        group (str): e.g., "9_months"
        condition (str): e.g., "toys"
        
    Returns:
        dict: Filtered dictionary {dyad_id -> {participant -> {channel -> data}}}
    """
    if group not in data:
        raise KeyError(f"Group {group} not found")

    filtered = {}
    for dyad_id, conds in data[group].items():
        if condition not in conds:
            continue

        cond_data = conds[condition]

        if participant:
            if participant in cond_data:
                filtered[dyad_id] = {participant: cond_data[participant]}

        else:

            raise KeyError(f"Participants not found")
        
    filtered = dict(sorted(filtered.items(), key=lambda x: int(x[0])))

    return filtered
