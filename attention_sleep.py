from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

features_val_path='data_valid_array.npy'
labels_train_path='label_train_array.npy'
labels_val_path='label_valid_array.npy'
labels_test_path= 'label_test_array.npy'
features_train_path = 'data_train_array.npy'
features_test_path =  'data_test_array.npy'

features_val= np.load(features_val_path)
labels_train= np.load(labels_train_path)
labels_val=np.load(labels_val_path)
labels_test = np.load(labels_test_path)
features_train =np.load(features_train_path)

features_test=np.load(features_test_path)
idx_label_val_ignore=np.where(labels_val==-1)
idx_label_train_ignore=np.where(labels_train==-1)
idx_label_test_ignore=np.where(labels_test==-1)
labels_val=np.delete(labels_val, idx_label_val_ignore)
features_val = np.delete(features_val,idx_label_val_ignore,axis=0)

labels_train=np.delete(labels_train, idx_label_train_ignore)
features_train = np.delete(features_train,idx_label_train_ignore,axis=0)

labels_test=np.delete(labels_test, idx_label_test_ignore)
features_test = np.delete(features_test,idx_label_test_ignore,axis=0)
idx_label_val_ignore=np.where(labels_val== 4)
idx_label_train_ignore=np.where(labels_train== 4)
idx_label_test_ignore=np.where(labels_test== 4)
labels_val=np.delete(labels_val, idx_label_val_ignore)
features_val = np.delete(features_val,idx_label_val_ignore,axis=0)

labels_train=np.delete(labels_train, idx_label_train_ignore)
features_train = np.delete(features_train,idx_label_train_ignore,axis=0)

labels_test=np.delete(labels_test, idx_label_test_ignore)
features_test = np.delete(features_test,idx_label_test_ignore,axis=0)
resampled_train_idx_zero=np.random.choice(*np.where(labels_train== 0),size=1272)
resampled_train_idx_one=np.random.choice(*np.where(labels_train== 1),size=1272)
resampled_train_idx_two=np.random.choice(*np.where(labels_train== 2),size=1272)
resampled_train_idx_three=np.random.choice(*np.where(labels_train== 3),size=1272)
resampled_train_idx_five=np.random.choice(*np.where(labels_train== 5),size=1272)
resampled_train_idx = np.concatenate((resampled_train_idx_zero,resampled_train_idx_one,resampled_train_idx_two,resampled_train_idx_three,resampled_train_idx_five))
resampled_train_idx
labels_train=labels_train[resampled_train_idx]
features_train = features_train[resampled_train_idx]
resampled_val_idx_zero = np.random.choice(*np.where(labels_val== 0),size=296)
resampled_val_idx_one = np.random.choice(*np.where(labels_val== 1),size=296)
resampled_val_idx_two = np.random.choice(*np.where(labels_val== 2),size=296)
resampled_val_idx_three = np.random.choice(*np.where(labels_val== 3),size=296)
resampled_val_idx_five = np.random.choice(*np.where(labels_val== 5),size=296)
resampled_val_idx = np.concatenate(
    (
        resampled_val_idx_zero,
        resampled_val_idx_one,
        resampled_val_idx_two,
        resampled_val_idx_three,
        resampled_val_idx_five
        
    
))
labels_val=labels_val[resampled_val_idx]
features_val = features_val [resampled_val_idx]
labels_val[np.where(labels_val== 5)]=4
labels_train[np.where(labels_train== 5)]=4
labels_test[np.where(labels_test == 5)] = 4
labels_val=labels_val.astype(np.int64)
labels_train=labels_train.astype(np.int64)
labels_test=labels_test.astype(np.int64)
print(labels_test)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=64, kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = (5100,1)

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    features_train,
    labels_train,
    validation_split=0.2,
    epochs=150,
    batch_size=20,
    callbacks=callbacks,
)

model.evaluate(features_test, labels_test, verbose=1)