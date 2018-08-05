import cv2                 # working with, mainly resizing, images
import numpy as np
import os
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training
from tqdm import tqdm      # a nice percentage bar ;)

TRAIN_DIR = 'train'
TEST_DIR = 'test'

def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'antelope': return 1
    elif word_label == 'bat': return 2
    elif word_label == 'beaver': return 3
    elif word_label == 'bobcat': return 4
    elif word_label == 'buffalo': return 5
    elif word_label == 'chihuahua': return 6
    elif word_label == 'chimpanzee': return 7
    elif word_label == 'collie': return 8
    elif word_label == 'dalmatian': return 9
    elif word_label == 'german+shepherd': return 10
    elif word_label == 'grizzly+bear': return 11
    elif word_label == 'hippopotamus': return 12
    elif word_label == 'horse': return 13
    elif word_label == 'killer+whale': return 14
    elif word_label == 'mole': return 15
    elif word_label == 'moose': return 16
    elif word_label == 'mouse': return 17
    elif word_label == 'otter': return 18
    elif word_label == 'ox': return 19
    elif word_label == 'persian+cat': return 20
    elif word_label == 'raccoon': return 21
    elif word_label == 'rat': return 22
    elif word_label == 'rhinoceros': return 23
    elif word_label == 'seal': return 24
    elif word_label == 'siamese+cat': return 25
    elif word_label == 'spider+monkey': return 26
    elif word_label == 'squirrel': return 27
    elif word_label == 'walrus': return 28
    elif word_label == 'weasel': return 29
    elif word_label == 'wolf': return 30


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data = process_test_data()