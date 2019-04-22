#------------------------------------------------------------------------------------------------
#import libraries
from __future__ import division
import numpy as np
import os
import glob
from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD

#------------------------------------------------------------------------------------------------
#create the dictionary to map writer and writing
d = {}
forms = pd.read_csv('forms_for_parsing.txt', header=None)
with open('forms_for_parsing.txt') as f:
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer
print(len(d.keys()))

#------------------------------------------------------------------------------------------------
tmp = []
target_list = []

path_to_files = os.path.join('data_subset', '*')
for filename in sorted(glob.glob(path_to_files)):
    # print(filename)
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    file, ext = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0] + '-' + parts[1]
    for key in d:
        if key == form:
            target_list.append(str(d[form]))

img_files = np.asarray(tmp)
img_targets = np.asarray(target_list)
print(img_files.shape)
print(img_targets.shape)

#for filename in img_files[:3]:
    # img = mpimg.imread(filename)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img, cmap='gray')
    # plt.show()

#------------------------------------------------------------------------------------------------
#encode the label
encoder = LabelEncoder()
encoder.fit(img_targets)
encoded_Y = encoder.transform(img_targets)
#print(img_files[:5], img_targets[:5], encoded_Y[:5])

#------------------------------------------------------------------------------------------------
#spliting into three data sets for cross validation by ratio of 7:1.5:1.5
train_files, rem_files, train_targets, rem_targets = train_test_split(
    img_files, encoded_Y, train_size=0.7, random_state=52, shuffle=True)

validation_files, test_files, validation_targets, test_targets = train_test_split(
    rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

print(train_files.shape, validation_files.shape, test_files.shape)
print(train_targets.shape, validation_targets.shape, test_targets.shape)


#------------------------------------------------------------------------------------------------
# Generator function for generating random crops from each sentence
batch_size = 16
num_classes = 50

def generate_data(samples, target_files, batch_size=batch_size, factor=0.1):
    num_samples = len(samples)
    from sklearn.utils import shuffle
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            batch_targets = target_files[offset:offset + batch_size]

            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]
                cur_height = im.size[1]

                height_fac = 113 / cur_height
                new_width = int(cur_width * height_fac)
                size = new_width, 113

                imresize = im.resize((size), Image.ANTIALIAS)
                now_width = imresize.size[0]

                avail_x_points = list(range(0, now_width - 113))
                pick_num = int(len(avail_x_points) * factor)
                random_startx = sample(avail_x_points, pick_num)

                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start + 113, 113))
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)

            X_train = np.array(images)
            y_train = np.array(targets)

            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            X_train = X_train.astype('float32')
            X_train /= 255

            #convert a class vector to binary class matric
            y_train = to_categorical(y_train, num_classes)
            yield shuffle(X_train, y_train)


train_generator = generate_data(train_files, train_targets, batch_size=batch_size, factor=0.3)
validation_generator = generate_data(validation_files, validation_targets, batch_size=batch_size, factor=0.3)
test_generator = generate_data(test_files, test_targets, batch_size=batch_size, factor=0.1)

#------------------------------------------------------------------------------------------------
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image, [56, 56])

#------------------------------------------------------------------------------------------------
#Modeling
row, col, ch = 113, 113, 1

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))

# Resize data within the neural network
model.add(Lambda(resize_image))  # resize images to allow for easy computation

# CNN model - Building the model suggested in paper

model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))
model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(512, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(256, name='dense2'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, name='output'))
model.add(Activation('softmax'))  # softmax since output is within 50 classes

model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
model.save_weights('low_loss.hdf5')
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
print(model.summary())

#------------------------------------------------------------------------------------------------
#Checkpoint
nb_epoch = 8
samples_per_epoch = 1000
nb_val_samples = 800


from keras.callbacks import ModelCheckpoint
filepath = "checkpoint2/check-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=1,
                                     callbacks=callbacks_list)


#------------------------------------------------------------------------------------------------
#Print accuracy
model.load_weights('low_loss.hdf5')
#train_scores = model.evaluate_generator(train_generator, 800)
test_scores = model.evaluate_generator(test_generator, 800)
#val_scores = model.evaluate_generator(validation_generator, 800)
print("Accuracy = ", test_scores[1])

# #Plot training and validation accuracy
# plt.plot(nb_epoch, train_scores[1], 'bo', label='Training Accuracy')
# plt.plot(nb_epoch, val_scores[1], 'b', label='Validation Accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
#
# #Plot training and test accuracy
# plt.plot(nb_epoch, train_scores[1], 'bo', label='Training Accuracy')
# plt.plot(nb_epoch, test_scores[1], 'b', label='Test Accuracy')
# plt.title('Training and test accuracy')
# plt.legend()
# plt.figure()
#
# plt.plot(nb_epoch, train_scores[0], 'bo', label='Training loss')
# plt.plot(nb_epoch, val_scores[0], 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()