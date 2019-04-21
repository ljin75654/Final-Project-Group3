# README

## Data Subset
There are 4899 png image files in the data subset. 

## forms_for_parsing 
This dataset contains images of each handwritten sentence with the dash-separated filename format. The first field represents the test code, second the writer id, third passage id, and fourth the sentence id.

## Installation
The following packages should be installed before running the program:
from __future__ import division<br/>
import numpy as np<br/>
import os<br/>
import glob<br/>
import time<br/>
from random import * <br/>
from PIL import Image<br/>
from keras.utils import to_categorical<br/>
from sklearn.preprocessing import LabelEncoder<br/>
from sklearn.model_selection import train_test_split<br/>
from keras.utils import plot_model<br/>
import matplotlib.pyplot as plt<br/>
import pandas as pd<br/>
import matplotlib.image as mpimg<br/>
from keras.models import Sequential<br/>
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization<br/>
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D<br/>
from keras.preprocessing.image import ImageDataGenerator<br/>
from keras.optimizers import SGD, Adam, RMSprop<br/>
import tensorflow as tf<br/>
from keras.callbacks import ModelCheckpoint<br/>
## 3_layer_CNN_Adam.py<br/>
3 convolution layers with Adam optimzer

## 3_layer_CNN_SGD.py<br/>
3 convolution layers with SGD optimizer

## 4_layer_CNN_Adam.py<br/>
4 convolution layers with Adam optimizer

## 4_layer_CNN_SGD.py<br/>
4 convolution layers with SGD optimizer


