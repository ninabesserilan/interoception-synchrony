from pathlib import Path
import pickle
pickle_rsa = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/RSA analysis/rsa_pickle.pkl')

with open(pickle_rsa, "rb") as f_rsa:
    rsa_data = pickle.load(f_rsa)


toys_rsa_data = rsa_data['toys']
notoys_rsa_data = rsa_data['no_toys']