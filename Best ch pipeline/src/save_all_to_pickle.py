import pickle

def save_all_final_data_pickle(
    toys_infant_final_dict,
    toys_mom_final_dict,
    no_toys_infant_final_dict,
    no_toys_mom_final_dict,
    output_path: str = "all_final_data.pkl"
):
    """
    Assemble all participant/condition final_dicts into a nested dict and save as a pickle.

    Parameters:
    - toys_infant_final_dict: final_dict for infant, toys
    - toys_mom_final_dict: final_dict for mom, toys
    - no_toys_infant_final_dict: final_dict for infant, no_toys
    - no_toys_mom_final_dict: final_dict for mom, no_toys
    - output_path: str, path to save pickle
    """
    all_final_data = {
        'toys': {
            'infant': toys_infant_final_dict,
            'mom': toys_mom_final_dict
        },
        'no_toys': {
            'infant': no_toys_infant_final_dict,
            'mom': no_toys_mom_final_dict
        }
    }

    # Save pickle
    with open(output_path, "wb") as f:
        pickle.dump(all_final_data, f)

    print(f"All data saved to {output_path}")
    return all_final_data
