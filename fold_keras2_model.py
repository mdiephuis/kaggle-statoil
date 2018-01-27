"""
    Train and prediction script for Statoil challenge
    Maurits Diephuis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Preprocessing functions
def normalize_bands(df):
    imgs = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        # Normalize values
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


# Model definition
def getModel2():
    # Building the model
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(75, 75, 3)))
    # Conv Layer 1
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    # Conv Layer 2
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    # Conv Layer 3
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    # Conv Layer 4
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    # Flatten the data for upcoming dense layers
    model.add(Flatten())

    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.4))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.4))

    # Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    mypotim = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='binary_crossentropy', optimizer=mypotim, metrics=['accuracy'])

    return model


# Call back functions
def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

    return [es, mcp_save, reduce_lr_loss]


# Main script start


# Load the json data in to panda dataframes
train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")

# Build Train and test datasets
X_train = normalize_bands(train)
X_test = normalize_bands(test)

# Get target labels from the train panda
target_train = train['is_iceberg']

# Train and fit the model
#
train_gen = ImageDataGenerator(
    rotation_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
)

val_gen = ImageDataGenerator()


n_folds = 15
batch_size = 35

for fold in range(n_folds):
    print('Run fold: {}'.format(fold))

    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, train_size=0.75)

    train_flow = train_gen.flow(X_train_cv, y_train_cv, batch_size=batch_size)
    validation_flow = val_gen.flow(X_valid, y_valid, batch_size=batch_size)

    file_path = 'fold_{}_model2_wts.hdf5'.format(fold)
    callbacks = get_callbacks(filepath=file_path, patience=35)

    model = getModel2()

    model.fit_generator(train_flow, epochs=120, verbose=0,
                        validation_data=validation_flow, callbacks=callbacks)

    model.load_weights(filepath=file_path)
    score = model.evaluate_generator(validation_flow)
    print('Fold run', fold)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_test = model.predict_proba(X_test)

    y_test = y_test.reshape((y_test.shape[0]))

    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = y_test
    submission.to_csv('fold_data/keras_fold_{}_submission.csv'.format(fold), index=False)
