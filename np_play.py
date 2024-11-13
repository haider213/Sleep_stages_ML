
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential

import numpy as np

features_val= np.load('features_val.npy')
labels_train= np.load('labels_train.npy')
labels_val=np.load('labels_val.npy')
labels_test = np.load('labels_test.npy')
features_train =np.load('features_train.npy',)

features_test=np.load('features_test.npy')

print("Shape of input data (X_train):", features_train.shape)

# Check the shape of labels
print("Shape of labels (y_train):", labels_train.shape)

model_5498603_sleep = Sequential()

model_5498603_sleep.add(Flatten(input_shape=(850,4)))
model_5498603_sleep.add(Dense(units=64, activation = 'relu'))
model_5498603_sleep.add(Dense(units= 64, activation = 'relu'))
model_5498603_sleep.add(Dense(units=7,activation='softmax'))
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