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
          print("ru")
        elif answer == 1:
          print("a")
        elif answer == 2:
          print("Aa")
        elif answer == 3:
          print("i")
        elif answer == 4:
          print("I")
        elif answer == 5:
          print("u")
        elif answer == 6:
          print("U")
        elif answer == 7:
          print("e")
        elif answer == 8:
          print("ai")
        elif answer == 9:
          print("o")
        elif answer == 10:
          print("au")
        elif answer == 11:
          print("am")
        elif answer == 12:
          print("ah")
        elif answer == 13:
          print("ka")
        elif answer == 14:
          print("kha")
        elif answer == 15:
          print("g")
        elif answer == 16:
          print("gh")
        elif answer == 17:
          print("ch")
        elif answer == 18:
          print("chh")
        elif answer == 19:
          print("j")
        elif answer == 20:
          print("jh")
        elif answer == 21:
          print("T")
        elif answer == 22:
          print("Th")
        elif answer == 23:
          print("D")
        elif answer == 24:
          print("Dh")
        elif answer == 25:
          print("N")
        elif answer == 27:
          print("th")
        elif answer == 28:
          print("d")
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