# README
## Introduction

This project addresses an image classification problem, and further application of the image classification explored in class. The IAM Handwriting Dataset chosen for this project is comprised of a collection of handwritten passages by various writers. The goal of this project is to use deep learning to classify the writers by their writing styles. The results of this project can be used in future applications, such as identifying criminals by signature in fraudulent cases. This report will discuss the dataset, network and training algorithm, experiment setup, results, and conclusions drawn from this analysis.

## Requirements 
This program runs on GPU. 

## Installation

The following packages should be installed before running the code:

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
