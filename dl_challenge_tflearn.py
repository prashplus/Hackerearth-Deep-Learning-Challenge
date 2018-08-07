# Imports and Headers
import cv2                 # working with, mainly resizing, images
import numpy as np
import os
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training
from tqdm import tqdm      # a nice percentage bar ;)
import pickle

import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import to_categorical


TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 150
LR = 1e-3

MODEL_NAME = 'dlchallenge_tflearn1.model'


# Data Preprocessing
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

# Data Spliting
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

# CNN building
tf.reset_default_graph()
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu')

network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')

network = conv_2d(network, 64, 3, activation='relu')

network = max_pool_2d(network, 2)

network = fully_connected(network, 512, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 30, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(x_train, y_train, n_epoch=100, shuffle=True, validation_set=(x_test, y_test),
          show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

'''
# Testing
import matplotlib.pyplot as plt
# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data.npy')



fig=plt.figure()
for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
'''