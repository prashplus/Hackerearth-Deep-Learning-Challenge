# Hackerearth Deep Learning Challenge May-2018

Hackerearth is known for its popular coding competitions and hackethons. In 2018 May, a Deep learning challenge was posted https://www.hackerearth.com/challenge/competitive/deep-learning-beginner-challenge/.

The challenge task was to create a Machine Learning model with the highest accuracy by using the provided image dataset (19000 images : 6.7GB) of 30 types of animals for both Training and Testing.

I have used **Convolution Neural Network** as the Deep learning model for the image classification. Trained on a well known ML library **Tensorflow** and the famous high level APIs **Keras** and **TFLearn** over the Tensorflow (for reducing the code complexity).

So, I have divided the task into simple 3 steps:

1. Data preprocessing

2. Training

3. Testing

Ofcourse, these can be further divided but for the shake of the simplification to all the developers I have kept like this. The dataset links are provided so no need to worry about it. Even if you are not having high-end GPUs, Google is providing free GPUs for running Jupyter Notebooks at https://colab.research.google.com.

## Data Preprocessing

All the dataset provided by the Hackerearth were not named or grouped according to the class labels i.e., Animal names, they had provided other meta data files like csv which contained all the labels respective to the image file name. Therefore, I have already preprocessed/renamed them by appending the animal name with the image file name which will make our task easy while assigning the labels to the images without looking at the meta files. I have uploaded the renamed data in my google drive. All you need to do is just download the data and extract it.

* Link for Datasets: https://drive.google.com/open?id=15amrCu1Xbte3gHpDvS0UBwkGzUynAOYQ

* Link for meta data: https://drive.google.com/open?id=1yXzNnxZQx5Npoi3Gs8XMtCPIN1YZcTYr

If its too large to download the file and don't want to waste your internet, then just download the preprocessed dataset, link is give at the end of this section.

### Now, its time for coding : -

The below steps are just the explanations of the code provided in the "data preprocessing.py" file.

1. First step is to parse through all the training images.

2. Extract the animal name from the file name to use it as the label for classification later.

3. Since, the images are in the RGB format that means for every pixel there are 3x8 bit values individiually for each color. But, by processing all the colors it will make our neural network complex. But, we can do is convert them into grayscale images so that each image can be represented in single 2D matrix, since black and white images can be represented by just 1s and 0s.

4. Resizing of the image, this step is essential since training on bigger images will result in to requirement of bigger Neural Network and lots-a processing and we don't need that for just 30 Labels.

5. Finally, coupling of the image with its respective label i.e., animal name and appending it in the numpy array.

Now, we are ready for the training part. But, we have to save the preprocessed data, since we are going to run the program multiple times and we don't want all those image conversions (time consuming stuffs) happen again and again. So, I have already uploaded the preprocessed dataset if anyone wants to skip this step.

Preprocessed data: https://drive.google.com/open?id=1Yzx5AeuNq7mo0AI3y1jf2SBi4gaZcFKE

## Model Design and Training

For starters just copy the CIFAR 10 Neural Net design and then make the changes to it. So, that you can learn by changing which parameter you can get the max efficiency or higher accuracy with lower time on training.

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

For training, we have to split the dataset in to two parts: Training and testing. It is preferred to divide the dataset into 70:30 ratio for training and testing respectively.

Now run the [Keras code](https://github.com/prashplus/Hackerearth-Deep-Learning-Challenge/blob/master/dl_challenge_keras.py) or [tflearn code](https://github.com/prashplus/Hackerearth-Deep-Learning-Challenge/blob/master/dl_challenge_tflearn.py).

If you don't have GPUs to train the model, then don't think about doing it on your machine because it may take months on doing so. I have included the [Jupyter Notebooks](https://github.com/prashplus/Hackerearth-Deep-Learning-Challenge/tree/master/Jupyter%20Notebooks) which you can copy and run it on the [Google Colab](https://colab.research.google.com). They are GPU runtime for Notebooks which provides Nvidia Tesla K80 for limited time. (Note: Don't close browser or it will terminate the sessions with the notebooks. And the training can't continue from where you left.)

Suggestions: Design your program in such a way that you put checkpoints on the way during the check points (saving your model's intermediate state) and do the training in batches instead of loading all the images at once.


NOTE: This repository is a work in progress project. The models will be more efficient over time.

## Author

* **Prashant Piprotar** - - [Prash+](https://github.com/prashplus)
and visit my blog [Nimbus](http://prashplus.blogspot.com) for more Tech Stuff
### http://www.prashplus.com