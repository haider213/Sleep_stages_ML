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

encoder = OneHotEncoder()


#reading data for one person
subject_id='5498603'
path_to_5498603_acc = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/motion/5498603_acceleration.txt'
participant_5498603_acc=pd.read_csv(path_to_5498603_acc,sep=' ')
print(participant_5498603_acc.head())
participant_5498603_acc.columns= ['time_to_sleep','ax','ay','az']

participant_5498603_acc=participant_5498603_acc[participant_5498603_acc['time_to_sleep']>=0]
path_to_5498603_ibi = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/heart_rate/5498603_heartrate.txt'
participant_5498603_ibi=pd.read_csv(path_to_5498603_ibi,sep=',')
participant_5498603_ibi.columns = ['time_to_sleep','ibi']
participant_5498603_ibi=participant_5498603_ibi[participant_5498603_ibi['time_to_sleep']>=0]
path_to_5498603_labels = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/labels/5498603_labeled_sleep.txt'
participant_5498603_labels=pd.read_csv(path_to_5498603_labels,sep=' ')

participant_5498603_labels.columns=['time_to_sleep','sleep_stage']
participant_5498603_labels['time_to_sleep'] = participant_5498603_labels['time_to_sleep'].astype('float64')


merged_dataframe = pd.merge_asof( participant_5498603_acc,participant_5498603_ibi,on='time_to_sleep')
merged_dataframe = pd.merge_asof(merged_dataframe ,participant_5498603_labels,on='time_to_sleep')

scaler = StandardScaler()

merged_dataframe[['ax','ay','az','ibi']]=scaler.fit_transform(merged_dataframe[['ax','ay','az','ibi']])
merged_dataframe=merged_dataframe[merged_dataframe['time_to_sleep']>30]


features = merged_dataframe[['ax','ay','az','ibi']].values
labels=merged_dataframe['sleep_stage'].values
print(f'Type of Labels {type(labels)}')
labels = labels[::1500]
print(f'number of samples: {len(labels)}')

num_samples=len(merged_dataframe)//1499
print(f'num_samples: {num_samples}')
num_timesteps= 1499 #number of data points for all sensors in each epoch of 30 seconds
num_features=features.shape[1] #number of features 

num_classes = len(merged_dataframe['sleep_stage'].unique())
labels_onehot=np.zeros((num_samples,num_classes))
labels_onehot=encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

features_3d = features[:num_samples*num_timesteps].reshape(num_samples,num_timesteps,num_features)
features_train, features_test, labels_train, labels_test = train_test_split(
    features_3d, labels_onehot, test_size=0.2, random_state=42)
# Split the data into training and testing sets


# Split the training data into training and validation sets
features_train, features_val, labels_train, labels_val = train_test_split(
    features_train, labels_train, test_size=0.2, random_state=42)
print("Shape of input data (X_train):", features_train.shape)

# Check the shape of labels
print("Shape of labels (y_train):", labels_train.shape)
"""
for i in range(labels):
    
    feature_3d[

data_combined = {'features':features_3d
                }
dataset = Dataset.from_dict(data_combined)
dataset.labels=labels
dataset.train_test_split(test_size=0.3)

print(dataset)


print(features_3d)
"""


model_5498603_sleep = Sequential()

model_5498603_sleep.add(Flatten(input_shape=(1499,4)))
model_5498603_sleep.add(Dense(units=64, activation = 'relu'))
model_5498603_sleep.add(Dense(units= 64, activation = 'relu'))
model_5498603_sleep.add(Dense(units=6,activation='softmax'))
#model_5498603_sleep.add(Dense(units=6, activation='softmax'))
#model_5498603_sleep.add(Dense(units=6))

model_5498603_sleep.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model_5498603_sleep.summary()



# Train the model
history = model_5498603_sleep.fit(
    features_train, labels_train,
    epochs=10, batch_size=32,
    validation_data=(features_val, labels_val)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model_5498603_sleep.evaluate(features_test, labels_test)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')