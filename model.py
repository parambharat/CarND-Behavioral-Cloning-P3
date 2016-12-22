# coding: utf-8
"""
Python script to train a ConvNet model that uses images from the dashboard to
predict the steering angle. The script reads data from the "data" directory in
the project root and saves the model weights and model in "models" directory.
"""

import os
import json

from keras.layers import Convolution2D, Dense, Lambda, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import pandas as pd

from utils import data_generator, get_datadirs, data_dir, root, read_data


def get_generators(dataset, split=0.2, batch_size=250):
    """
    Utility function to create data generators for training and validation.
    Input:
        dataset, DataFrame: image locations and corresponding steering angles.
        split, float32: size of dataset reserved for validation.
        batch_size,int32: size of a single batch of training.
    Output:
        train_generator, __generator__: (batch_images, batch_steering)
        val_generator, __generator__: (batch_images, batch_steering)
    """
    train, val = train_test_split(dataset, test_size=split)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    train_generator = data_generator(train, batch_size=batch_size)
    val_generator = data_generator(val, batch_size=batch_size)
    return train_generator, val_generator


def get_model():
    """
    Utility function creates the keras model that will be trained to predict
    the steering angle based on the input image
    Input:
        None
    Output:
        model, keras.Model.model
    """
    imshape = (32, 32, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=imshape))
    model.add(Convolution2D(3, 1, 1))
    model.add(Convolution2D(64, 7, 7, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 4, 4, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model


def compile_model(model, lr=1e-4, loss='mse'):
    """
    Utility function to compile a keras model using AdamOptimizer.
    Input:
        model, keras.Model.model
        lr, float32: learning rate to be used during optimization
        loss, string: type of loss function to be used to train the model.
    Output:
        model, keras.Model.model compiled with optimizer and loss function.

    """
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=loss)

    return model


def train_model(model, train_generator, val_generator,
                checkpoint_file, num_epochs=10, num_samples=50000):
    """
    Utility function to train a pre-compiled keras model.
    Input:
        model, keras.Model.model: pre-compiled keras model.
        train_generator, __generator__: training data generator.
        val_generator, __generator__: validation data generator.
        checkpoint_file, string: path of intermediate checkpoint file.
        num_epochs, int32: number of training epochs to run.
        num_samples, int32: number of samples in each epoch
    Output:
        model, keras.Model.model: trained model.
        history, keras.History.history: keras model history object
    """
    checkpoint = ModelCheckpoint(
        checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto')

    reducelr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=1,
        verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    history = model.fit_generator(
        train_generator, samples_per_epoch=num_samples, nb_epoch=num_epochs,
        verbose=1, callbacks=[checkpoint, reducelr],
        validation_data=val_generator, nb_val_samples=10, max_q_size=16,
        nb_worker=8, pickle_safe=True)

    return model, history


def save_model(model, model_dir):
    """
    Utility to save model and model weights.
    Input:
        model, keras.Model.model
        model_dir, string: absolute path of the model directory.
    Output:
        None
    """
    weights_file = os.path.join(model_dir, 'model.h5')
    model_file = os.path.join(model_dir, 'model.json')
    model.save_weights(weights_file, True)
    with open(model_file, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print('Model Saved')


def main():
    dir_names = get_datadirs(data_dir)

    model_dir = os.path.join(root, 'models',)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_file = os.path.join(root, 'models', 'checkpoint')

    all_files = [read_data(f) for f in dir_names]
    data = pd.concat(all_files)

    train_generator, val_generator = get_generators(data)

    model = get_model()
    model = compile_model(model)
    model, _ = train_model(model, train_generator,
                           val_generator, checkpoint_file, num_epochs=10,
                           num_samples=50000)
    save_model(model, model_dir)


if __name__ == '__main__':
    main()
