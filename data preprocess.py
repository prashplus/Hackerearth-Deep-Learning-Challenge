from __future__ import print_function
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
    #img_count = 0
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        #img_count += 1
        #if img_count > 6000 : break

    shuffle(training_data)

    file = open('hackerearth_train_data_200x200.dat', 'wb')
    pickle.dump(training_data, file)  
    file.close()

    return training_data

train_data = create_train_data()