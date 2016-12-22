
# Behaviour Cloning

## Problem Statement

Train a deep neural network to drive a car like you! The team at Udacity have built a simulator for students to be able to drive and record data from the simulator. The data is collected in the form of a driving log and images. Images are recorded at 10hz with cameras in the left, center and right. Each set of images contains a corresponding steering angle in the driving. In this project we build a framework consisting of a Convolution Neural Network to predict the steering angle given the image of the road from the dashboard of a car.

## Requirements

To be able to successfully execute the code in this project you might require the following libraries, packages and  data.

1. [Anaconda Python 3.5](https://www.continuum.io/downloads)
2. [Keras](https://keras.io/)
3. [OpenCV](https://anaconda.org/menpo/opencv3)
4. [numpy](https://anaconda.org/anaconda/numpy)
5. [flask-socketio](https://anaconda.org/pypi/flask-socketio)
6. [eventlet](https://anaconda.org/conda-forge/eventlet)
7. [pillow](https://anaconda.org/anaconda/pillow)
8. [h5py](https://anaconda.org/conda-forge/h5py)
9. The simulator - [linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip), [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip), [windows 32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip), [windows 64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
10. [The data](https://drive.google.com/open?id=0B94J1XBB-7XKeUpSQ2JvTnIxblk)

## Project Structure

The below tree represents the project structure and all the necessary files and folders.
```
.
├── data/
├── drive.py
├── model.py
├── models
├── readme.md
├── simulator_files/
└── utils.py
```

Ensure that the data is present in the `data` folder with the following prefix `data` and contains the data from the simulator. For instance, if you have the dataset from udacity rename the directory as `data_udacity` it should have the following structure.

```
data
│
└── data_udacity
    ├── driving_log.csv
    └── IMG
```

## Data Analysis

All the images recorded by the simulator are in the *`.jpg`* format with the following dimensions *`160 X 320 X 3`*. The log file `driving_log.csv` contains the following columns `[center, left, right, steering, throttle, brake, speed]` the values of interest are `[center, left, right, steering]` the first three columns contain the paths of the image files usually in the `IMG` subdirectory and the `steering` column contains the steering angle.
It must be noted that the data was collected in this project by recording seperate instances of recovery driving coupled with the Udacity [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). The entire dataset can be downloaded from [here](https://drive.google.com/file/d/0B94J1XBB-7XKeUpSQ2JvTnIxblk/view?usp=sharing)
It was noticed that the steering angles were normalized between -1 and 1 with -1 indicating 25&deg; `left` and 1 indicating 25&deg; `right`. Furthermore, Most of the steering angles are zeros since the recording was done using a keyboard. This was however handled in the data pre-processing step to a small extent.

## Data Pre-Processing

As noticed during Data Analysis we have 3 images per steering angle `left`, `right` and `center`. The steering angle is recorded only for the `center` image. With deep learning any model is only as good as the data. The more consistent and uncorrelated data we have, the better our model will turn out to be. We must either collect hours of training data or augment our existing data to have more training samples.

One simple augmentation we could apply is  to use the `left` and `right` images as if they were the center images. To do this, it's assumed that the steering angles corresponding to the `left` and `right` images must be shifted by a small constant. We assume a constant 0f `0.25` (This was arrived at after taking values in the range `0.1` to `0.3` at intervals of `0.05`). We apply this to all the images and get two times more images. We also randomly flip the images that have non-zero steering angles with a chance of `50%` and negate the steering angle for the flipped images.

Additionally, we crop the image to remove `20%` of the image from the top and the bottom and downsample it to `32 x 32 x 3` this shape make the images more manageable and the neural network memory efficient to train and predict. While we would like to normalize the images at this point this is done in a layer in the neural network to take advantage of the GPU while training.

Most of these are performed in the `utils.py` module and imported in the `model.py` and `drive.py` modules.

## Implementation

Before proceeding with the implementation it must be noted that the training data is split into train and validation datasets using the sklearn's [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn-model-selection-train-test-split).

The data is generated for both training and validation sets using python generators and images are only read into memory during runtime. This makes the implementation scalable and resource efficient.

As discussed we implement a convolution neural network in this project. This is presented in the `model.py` module. At a high level the following are the details of the neural network.

0. Input shape = 32,32,3
1. Lambda normalization layer . Normalizes the image pixel values to between `-0.5` and `0.5`
2. 2D Convolution - filter_size = 1, num_filter = 3, stride=0, padding='same', activation=none.
3. 2D Convolution - filter_size = 7, num_filter = 64, stride=0, padding='valid', activation=elu.
4. 2D Max Pooling - pool size = 2
5. 2D Convolution - filter_size = 4, num_filter = 128, stride=0, padding='valid', activation=elu.
6. 2D Max Pooling - pool size = 2
7. 2D Convolution - filter_size = 4, num_filter = 256, stride=0, padding='valid', activation=elu.
8. 2D Max Pooling - pool size = 2
9. Dropout - probability 0.5
10. Fully Connected layer = 512 Neurons, activation=elu.
11. Dropout - probability 0.5
12. Fully Connected layer = 128 Neurons, activation=elu.
13. Dropout - probability 0.5
12. Fully Connected layer = 32 Neurons, activation=elu.
12. Fully Connected layer = 1 Neuron, activation=None.

The above model was built in keras. The loss is defined to be `mean squared error` and the model is compiled with the `Adam Optimizer` with a default learning rate of `1e-4` that drops when there is no change in the validation accuracy between epochs. [Checkpoints](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L220) are created using keras [callbacks](https://keras.io/callbacks/) and stores the models when the validation accuracy is at it's best.

The model is trained with a mini-batch-size of `250` at `50,000` samples per epoch for `10 epochs`. Any further increase in epochs did not prove to be highly beneficial. Considering that the time for running each epoch is approximately 20 seconds a few more epochs of training didn't hurt.

Finally, the model and its associated weights are stored in the `model` directory of the project root. The `model/model.json` is passed as an argument to the `drive.py` file while running it.

## Results

The final model performed reasonable well on the validation dataset with a final validation loss(mse) of `0.0297` The model was able to drive through the track without going off track and was able to generalize to track 2 to a large extent. The demostration of the model driving in track 1 can be found [here](https://youtu.be/pVGz8hXLQgY).
