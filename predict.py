import keras
import os
import numpy as np
import pickle
import cv2
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

IMG_SIZE = 150

def create_alientest_data():
    alien_test = []
    for i in os.listdir(test_path):
        img_loc = test_path+ "\\" + i
        print(img_loc)
        #try:
        img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img = cv2.bitwise_not(img)
        alien_test.append([np.array(img),np.array("NULL")])
        #except:
            #print("Error at :"+ img_loc)
    return alien_test

def labelling(result):
  #print(result)
  for i in range(result.shape[0]):
    #print(i)
    answer = 0
    for j in range(result[i].shape[0]):
      if(result[i][j]==1):
        #print(answer)
        if answer == 0:
          print("antelope")
        elif answer == 1:
          print("bat")
        elif answer == 2:
          print("beaver")
        elif answer == 3:
          print("bobcat")
        elif answer == 4:
          print("buffalo")
        elif answer == 5:
          print("chihuahua")
        elif answer == 6:
          print("chimpanzee")
        elif answer == 7:
          print("collie")
        elif answer == 8:
          print("dalmatian")
        elif answer == 9:
          print("germanshepherd")
        elif answer == 10:
          print("grizzlybear")
        elif answer == 11:
          print("hippopotamus")
        elif answer == 12:
          print("horse")
        elif answer == 13:
          print("killerwhale")
        elif answer == 14:
          print("mole")
        elif answer == 15:
          print("mouse")
        elif answer == 16:
          print("otter")
        elif answer == 17:
          print("ox")
        elif answer == 18:
          print("persiancat")
        elif answer == 19:
          print("raccoon")
        elif answer == 20:
          print("rat")
        elif answer == 21:
          print("rhinoceros")
        elif answer == 22:
          print("seal")
        elif answer == 23:
          print("siamesecat")
        elif answer == 24:
          print("spidermonkey")
        elif answer == 25:
          print("squirrel")
        elif answer == 26:
          print("walrus")
        elif answer == 27:
          print("weasel")
        elif answer == 28:
          print("wolf")
        else:
          print("Other")
      answer += 1

model_path = 'trained_model.h5'
model = load_model(model_path)
test_path = 'alien_test'

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train = create_alientest_data()
x_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
array = model.predict(x_train)
labelling(array)