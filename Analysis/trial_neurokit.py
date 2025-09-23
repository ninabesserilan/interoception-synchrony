import numpy as np
import pandas as pd
import neurokit2 as nk
import pathlib
import numpy as np
# Generate synthetic signals
# ecg = nk.ecg_simulate(duration=10, heart_rate=70)
# ppg = nk.ppg_simulate(duration=10, heart_rate=70)
# rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
# eda = nk.eda_simulate(duration=10, scr_number=3)
# emg = nk.emg_simulate(duration=10, burst_number=2)

# Visualise biosignals
# data = pd.DataFrame({"ECG": ecg,
#                      "PPG": ppg,
#                      "RSP": rsp,
#                      "EDA": eda,
#                      "EMG": emg})
# nk.signal_plot(data, subplots=True)




# data = pd.DataFrame({"ECG": ecg,
#                      "PPG": ppg,
#                      "RSP": rsp,
#                      "EDA": eda,
#                      "EMG": emg})
# nk.signal_plot(data, subplots=True)


# infant_04_9_wtoys_ch0 = np.loadtxt(pathlib.Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception synchrony/ibis_wp3_004_wp3_wtoys_smarting_ECG1_channel0_500hz.csv'))
# infant_04_9_wtoys_ch1 = np.loadtxt(pathlib.Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception synchrony/ibis_wp3_004_wp3_wtoys_smarting_ECG1_channel1_500hz.csv'))
# infant_04_9_wtoys_ch2 = np.loadtxt(pathlib.Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception synchrony/ibis_wp3_004_wp3_wtoys_smarting_ECG1_channel2_500hz.csv'))


# data_infant_04_9_wtoys_ch0 = pd.DataFrame({"ch0 - should be best": infant_04_9_wtoys_ch0} )
# data_infant_04_9_wtoys_ch1 = pd.DataFrame({"ch1 - should be bad": infant_04_9_wtoys_ch1} )
# data_infant_04_9_wtoys_ch2 = pd.DataFrame({"ch2 - should be ok": infant_04_9_wtoys_ch2} )

# nk.signal_plot(data_infant_04_9_wtoys_ch0)
# nk.signal_plot(data_infant_04_9_wtoys_ch1)
# nk.signal_plot(data_infant_04_9_wtoys_ch2)


