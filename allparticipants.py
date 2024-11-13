import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import glob

path_to_dataset = "/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/data_compiled.csv"

path_to_labels = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/labels'

participants= [file.split('/')[10].split('_')[0] for file in glob.glob(path_to_labels+'/**/*.txt', recursive=True) ]
print(participants)

               
              
dataset= pd.read_pickle('correct_label.pkl')
#dataset = dataset.drop(columns=[dataset.columns[0]],axis=1)

    



print(dataset.head())
print(dataset.columns)
print(dataset.info)
print(dataset.describe())
print(f'Sleep Stages: {dataset["sleep_stage"].unique()}')

print(f'Number of samples = {len(dataset)/850}')
#print(f'Number of NaN samples = {dataset.isnull().sum(axis=1)}')
print(f'Number of stage 1 samples = {len(dataset[dataset["sleep_stage"]==1])/(850)}')
print(f'Number of stage -1 samples = {len(dataset[dataset["sleep_stage"]==-1])/(850)}')
print(f'Number of stage 0 samples = {len(dataset[dataset["sleep_stage"]==0])/(850)}')
print(f'Number of stage 2 samples = {len(dataset[dataset["sleep_stage"]==2])/(850)}')
print(f'Number of stage 3 samples = {len(dataset[dataset["sleep_stage"]==3])/(850)}')
print(f'Number of stage 4 samples = {len(dataset[dataset["sleep_stage"]==4])/(850)}')
print(f'Number of stage 5 samples = {len(dataset[dataset["sleep_stage"]==5])/(850)}')