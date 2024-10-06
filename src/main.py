import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from logic.FilterSignals import bandpass_filter
import os

minfreq = 0.8
maxfreq = 2.0

nameFile = 'apollo12_catalog_GradeA_final'
path = 'src/catalog/apollo12_catalog_GradeA_final.csv'
file = pd.read_csv(path)

os.makedirs('results', exist_ok=True)
os.makedirs('plot', exist_ok=True)

row = file.iloc[15]
test_filename = row['filename'] 
arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')

print('File Name:' + test_filename)

data_directory = 'src/data/lunar/training/data/S12_GradeA/'
csv_file = f'{data_directory}{test_filename}.csv'
data_fileTest = pd.read_csv(csv_file)

csv_times = np.array(data_fileTest['time_rel(sec)'].tolist())
csv_data = np.array(data_fileTest['velocity(m/s)'].tolist())

fs = 1 / np.mean(np.diff(csv_times))
csv_data_filt = bandpass_filter(csv_data, minfreq, maxfreq, fs)

sta_len = 120
lta_len = 600
characteristicFunctionTrigger = classic_sta_lta(csv_data_filt, int(sta_len * fs), int(lta_len * fs))

thr_on = 4.8
thr_off = 1.5
on_off = np.array(trigger_onset(characteristicFunctionTrigger, thr_on, thr_off))

print(on_off)
detection_times = []
fnames = []
max_values = []
min_values = []

for i in np.arange(0, len(on_off)):
    triggers = on_off[i]
    on_time = arrival_time + timedelta(seconds=csv_times[triggers[0]])
    on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
    detection_times.append(on_time_str)
    fnames.append(test_filename)

    trigger_start = triggers[0] - int(sta_len * fs) // 2
    trigger_end = triggers[0] + int(sta_len * fs) // 2
    trigger_start = max(trigger_start, 0)
    trigger_end = min(trigger_end, len(csv_data_filt))  

    max_values.append(np.max(csv_data_filt[trigger_start:trigger_end]))
    min_values.append(np.min(csv_data_filt[trigger_start:trigger_end]))

detect_df = pd.DataFrame(data={
    'filename': fnames,
    'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
    'time_rel(sec)': csv_times[on_off[:, 0]],
    'max_velocity(m/s)': max_values,
    'min_velocity(m/s)': min_values
})

detect_df.to_csv(f'results/detect_1.csv', index=False)

fig, axs = plt.subplots(3, 1, figsize=(12, 18))

axs[0].plot(csv_times, csv_data, label='Original Data', color='blue')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Velocity (m/s)')
axs[0].set_title('Original Velocity Data', fontweight='bold')
axs[0].grid()
axs[0].legend()

axs[1].plot(csv_times, csv_data_filt, label='Filtered Data', color='green')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Filtered Velocity (m/s)')
axs[1].set_title('Filtered Velocity Data', fontweight='bold')
axs[1].grid()
axs[1].legend()

axs[2].plot(csv_times, characteristicFunctionTrigger, label='CFT', color='orange')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('CFT')
axs[2].set_title('Characteristic Function (CFT)', fontweight='bold')
axs[2].grid()
axs[2].legend()

for i in np.arange(0, len(on_off)):
    triggers = on_off[i]
    axs[1].axvline(x=csv_times[triggers[0]], color='red', label='Trig. On' if i == 0 else "")
    axs[1].axvline(x=csv_times[triggers[1]], color='purple', label='Trig. Off' if i == 0 else "")

plt.tight_layout()
plt.savefig(f'plot/combined_plots_with_triggers_results_presentation.png')
plt.close(fig)
