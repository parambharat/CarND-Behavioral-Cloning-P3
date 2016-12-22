# coding: utf-8
"""
Python script contains all the necesary utility functions required in this
project
"""


import math
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob


def get_datadirs(data_dir):
    """
    Utility to get data directories with prefix "data_" containing data
    recorded from various training sesions
    Input:
        data_dir, string - path of the data directory
    Output:
        paths, list - paths of individual data directories
    """
    paths = []
    for dirname in glob(os.path.join(data_dir, 'data*')):
        log_file = os.path.join(dirname, 'driving_log.csv')
        if os.path.isfile(log_file):
            paths.append(dirname)
    return paths


def get_fname(f_path):
    """
    Utility to get the file name from a given path string.
    Input:
        f_path, string: path of the file. (relative or absolute)
    Output:
        f_name, string: name of the file.
    """
    _, f_name = os.path.split(f_path)
    return f_name


def read_data(log_dir):
    """
    Utility function to fetch and clean up data.
        1. It load the data from the csv file.
        2. Finds the absolute paths of the files in the IMG directory.
        3. Manipulates the left and right steering angles.
    Input:
        log_dir, string: The absolute path of the log directory
    Output:
        DataFrame: Contains image path and the related output steering angle
    """
    log_file = os.path.join(log_dir, 'driving_log.csv')
    columns = ['center', 'left', 'right',
               'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(log_file, names=columns)

    paths = data[['center', 'left', 'right']]
    paths = paths.applymap(lambda x: x.strip())
    paths = paths.applymap(lambda x: get_fname(x))
    paths = paths.applymap(lambda x: os.path.join(log_dir, 'IMG', x))
    data[['center', 'left', 'right']] = paths

    necessary_cols = ['center', 'left', 'right', 'steering']
    final_cols = ['image', 'steering']

    all_data = data[necessary_cols]

    center_data = pd.DataFrame(all_data[['center', 'steering']])
    center_data.columns = final_cols

    left_data = pd.DataFrame(all_data[['left', 'steering']])
    left_data['steering'] += 0.25
    left_data.columns = final_cols

    right_data = pd.DataFrame(all_data[['right', 'steering']])
    right_data['steering'] -= 0.25
    right_data.columns = final_cols

    stacked = [left_data, center_data, right_data]
    final_data = pd.concat(stacked).reset_index(drop=True)
    final_data['steering'] = final_data['steering'].astype(np.float32)
    final_data['image'] = final_data['image'].apply(lambda x: x.strip())

    return final_data


def read_image(image_file):
    """
    Utility to read and image file as an ndarray
    Input:
        image_file, string: Absolute path of the image file
    Output:
        Image, ndarray: shape = (num_rows, num_cols, num_channels)
    """
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize(image):
    """
    Utility to crop and reshape the image to input it into the network.
    Input:
        Image, ndarray: shape = (num_rows, num_cols, num_channels)
    Output:
        image, ndarray: shape = (32,32,3)
    """
    imshape = image.shape
    top_crop, bot_crop = math.floor(imshape[0]*0.2), math.floor(imshape[0]*0.8)
    image = image[top_crop:bot_crop, 0:imshape[1]]
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)


def flip_image(image, steering):
    """
    Utility to filp an image and its corresponding steering angle with
    a 50% chance
    Input:
        image, ndarray: shape = (num_rows, nun_cols, num_channels).
        steering, float32: steering angle corresponding to input image.
    Output:
        image, ndarray: shape = (num_rows, nun_cols, num_channels).
        steering, float32: The negative of the steering input
                           if image is flipped.
    """
    toss = np.random.randint(0, 2)
    if toss == 0:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def data_generator(df, batch_size=64):
    """
    Utility to generate images and corresponding steering angles infinitely.
        1. Shuffles the dataset
        2. Reads the images from the image path.
        3. Resizes the image. (Cropping is also done here)
        4. If image's steering angle is not 0 then flip image with a 50% chance
    Input:
        df, DataFrame: Must contain two cols 'image' and 'steering'
        batch_size, int = size of an individual batch
    Output:
        images, ndarray: shape=(batch_size, num_rows, num_cols, num_channels),
                         dtype=np.uint8
        steering_angles, ndarray: shape=(batch_size), dtype=np.float32
    """
    while True:
        idx = df.index.values
        np.random.shuffle(idx)
        df = df.iloc[idx]
        batch_images, batch_steers = [], []
        for i in np.random.choice(idx, size=batch_size, replace=False):
            image_file, y_steer = df.iloc[i]['image'], df.iloc[i]['steering']
            image = read_image(image_file)
            image = resize(image)
            if y_steer != 0:
                image, y_steer = flip_image(image, y_steer)
            batch_images.append(image), batch_steers.append(y_steer)
        yield np.array(batch_images), np.array(batch_steers)


root = os.getcwd()
data_dir = os.path.join(root, 'data')


def main():
    dir_names = get_datadirs(data_dir)
    total_data = pd.concat([read_data(f) for f in dir_names])
    print(total_data.head())


if __name__ == '__main__':
    main()
