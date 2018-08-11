from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import os

import cv2                 # working with, mainly resizing, images
import numpy as np
import os
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training
from tqdm import tqdm      # a nice percentage bar ;)
import pickle

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 150
LR = 1e-3

batch_size = 32
num_classes = 30
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_prashplus_trained_model.h5'


def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'antelope': return 0
    elif word_label == 'bat': return 1
    elif word_label == 'beaver': return 2
    elif word_label == 'bobcat': return 3
    elif word_label == 'buffalo': return 4
    elif word_label == 'chihuahua': return 5
    elif word_label == 'chimpanzee': return 6
    elif word_label == 'collie': return 7
    elif word_label == 'dalmatian': return 8
    elif word_label == 'german+shepherd': return 9
    elif word_label == 'grizzly+bear': return 10
    elif word_label == 'hippopotamus': return 11
    elif word_label == 'horse': return 12
    elif word_label == 'killer+whale': return 13
    elif word_label == 'mole': return 14
    elif word_label == 'moose': return 15
    elif word_label == 'mouse': return 16
    elif word_label == 'otter': return 17
    elif word_label == 'ox': return 18
    elif word_label == 'persian+cat': return 19
    elif word_label == 'raccoon': return 20
    elif word_label == 'rat': return 21
    elif word_label == 'rhinoceros': return 22
    elif word_label == 'seal': return 23
    elif word_label == 'siamese+cat': return 24
    elif word_label == 'spider+monkey': return 25
    elif word_label == 'squirrel': return 26
    elif word_label == 'walrus': return 27
    elif word_label == 'weasel': return 28
    elif word_label == 'wolf': return 29


def create_train_data():
    training_data = []

    if os.path.exists("train_data.dat"):
        file = open('train_data.dat', 'rb')
        training_data = pickle.load(file)
        file.close()
        return training_data

    img_count = 0
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        img_count += 1
        #if img_count > 6000 : break

    shuffle(training_data)

    file = open('train_data.dat', 'wb')
    pickle.dump(training_data, file)  
    file.close()

    return training_data

train_data = create_train_data()

train = train_data[:-3000]
test = train_data[-3000:]

# Training Data
x_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train = [i[1] for i in train]

y_train = to_categorical(y_train,30)

# Testing Data
x_test = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

y_test = [i[1] for i in test]
print(y_test)
y_test = to_categorical(y_test,30)
print(y_test)

# CNN Model Arch
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Train the model using RMSprop
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])