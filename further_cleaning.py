
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
scaler = StandardScaler()


dataset=pd.read_pickle('correct_label.pkl')

data_train_array = np.empty((0, 5100))
label_train_array = np.empty((0,))

data_valid_array = np.empty((0, 5100))
label_valid_array = np.empty((0,))

data_test_array = np.empty((0, 5100))
label_test_array = np.empty((0,))

participant_training=[5498603, 2638030, 8530312, 1066528, 5132496, 4314139,
        781756, 7749105, 5797046, 8000685,
       3509524, 9961348, 2598705, 4018081, 8692923, 9618981, 6220552,
       1818471, 8173033, 1449548,
       3997827]
participant_validation = [4426783,  759667, 1455390,8686948,  844359,   46343]

participant_test= [1360686, 9106476]


for participant in participant_training:
    a=dataset[dataset['participant_x'] == participant].sort_values(by=['time_to_sleep'])
    
    a['cos_t'] = np.cos(a['time_to_sleep'])
    a['sin_t'] = np.cos(a['time_to_sleep'])
    a[['ax','ay','az','ibi','cos_t','sin_t']] = scaler.fit_transform(a[['ax','ay','az','ibi','cos_t','sin_t']])
    a_label=a['sleep_stage_y']
    for i in range(0, len(a), 850):
        chunk = a.iloc[i:i+850]
        flattened_chunk = chunk[['ax','ay','az','ibi','cos_t','sin_t']].values.flatten()
        #flattened_chunk = chunk.drop(columns='sleep_stage_y').values.flatten()
        data_train_array = np.vstack((data_train_array, flattened_chunk))  # Append flattened chunk to data array
        labels = chunk.iloc[-1]['sleep_stage_y']
        label_train_array = np.append(label_train_array, labels)

print("Data array shape:", data_train_array.shape)
print("Label array shape:", label_train_array.shape)

np.save('data_train_array.npy', data_train_array)
np.save('label_train_array.npy', label_train_array)

for participant in participant_validation:
    a=dataset[dataset['participant_x'] == participant].sort_values(by=['time_to_sleep'])
    
    a['cos_t'] = np.cos(a['time_to_sleep'])
    a['sin_t'] = np.cos(a['time_to_sleep'])
    a[['ax','ay','az','ibi','cos_t','sin_t']] = scaler.fit_transform(a[['ax','ay','az','ibi','cos_t','sin_t']])
    a_label=a['sleep_stage_y']
    for i in range(0, len(a), 850):
        chunk = a.iloc[i:i+850]
        flattened_chunk = chunk[['ax','ay','az','ibi','cos_t','sin_t']].values.flatten()
        #flattened_chunk = chunk.drop(columns='sleep_stage_y').values.flatten()
        data_valid_array = np.vstack((data_valid_array, flattened_chunk))  # Append flattened chunk to data array
        labels = chunk.iloc[-1]['sleep_stage_y']
        label_valid_array = np.append(label_valid_array, labels)

print("Data array shape:", data_valid_array.shape)
print("Label array shape:", label_valid_array.shape)

np.save('data_valid_array.npy', data_valid_array)
np.save('label_valid_array.npy', label_valid_array)


for participant in participant_test:
    a=dataset[dataset['participant_x'] == participant].sort_values(by=['time_to_sleep'])
    
    a['cos_t'] = np.cos(a['time_to_sleep'])
    a['sin_t'] = np.cos(a['time_to_sleep'])
    a[['ax','ay','az','ibi','cos_t','sin_t']] = scaler.fit_transform(a[['ax','ay','az','ibi','cos_t','sin_t']])
    a_label=a['sleep_stage_y']
    for i in range(0, len(a), 850):
        chunk = a.iloc[i:i+850]
        flattened_chunk = chunk[['ax','ay','az','ibi','cos_t','sin_t']].values.flatten()
        #flattened_chunk = chunk.drop(columns='sleep_stage_y').values.flatten()
        data_test_array = np.vstack((data_test_array, flattened_chunk))  # Append flattened chunk to data array
        labels = chunk.iloc[-1]['sleep_stage_y']
        label_test_array = np.append(label_test_array, labels)

print("Data array shape:", data_test_array.shape)
print("Label array shape:", label_test_array.shape)

np.save('data_test_array.npy', data_test_array)
np.save('label_test_array.npy', label_test_array)