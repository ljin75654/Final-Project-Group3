# README
## Introduction

This project addresses an image classification problem, and further application of the image classification explored in class. The IAM Handwriting Dataset chosen for this project is comprised of a collection of handwritten passages by various writers. The goal of this project is to use deep learning to classify the writers by their writing styles. The results of this project can be used in future applications, such as identifying criminals by signature in fraudulent cases. This report will discuss the dataset, network and training algorithm, experiment setup, results, and conclusions drawn from this analysis.

##Requirements 
This program runs on GPU. 

##Installation
###The following packages should be installed before running the code:
from __future__ import division
import numpy as np
import os
import glob
import time
from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
