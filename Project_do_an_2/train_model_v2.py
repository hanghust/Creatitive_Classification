# -*- coding: utf-8 -*-

import matplotlib as mpl
import keras
import cv2
import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from preprocessing import take_photos, prepare_image_test, prepare_image_train, prepare_image_val
import os
import pandas as pd 
import tensorflow as tf
from scipy.io import loadmat
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,SeparableConv2D,BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

def train_model():

    label_encoder = LabelEncoder()
    x_train, y_train = prepare_image_train()
    x_val, y_val = prepare_image_val()
    num_class = 4
    y_train = label_encoder.fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_train, num_class)
    y_val = label_encoder.fit_transform(y_val)
    y_val = keras.utils.to_categorical(y_val, num_class)



    # Base model without Fully connected Layers
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x=base_model.output
    # Add some new Fully connected layers to 
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x = Dropout(0.25)(x)
    x=Dense(512,activation='relu')(x) 
    x = Dropout(0.25)(x)
    preds=Dense(num_class, activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)
    model.summary()
    # dong bang tu layer 1:87
    # for i,layer in enumerate(model.layers):
    #     print("{}: {}".format(i,layer))
    for layer in model.layers[:87]:
        layer.trainable=False
    for layer in model.layers[87:]:
        layer.trainable=True

    epochs = 10
    learning_rate = 0.005
    decay_rate = learning_rate / epochs
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])



    datagen = ImageDataGenerator(
        featurewise_center=True,\
        featurewise_std_normalization=True,\
        rotation_range=15,\
        width_shift_range=0.2,\
        height_shift_range=0.2,\
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), steps_per_epoch=len(x_train) / 64 ,epochs=epochs,verbose=1,validation_data=(x_val,y_val))


    model_json = model.to_json()
    with open('model_cv/model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_cv/model.h5')

    #testing
    scores = model.evaluate(x_val, y_val, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def test_acc():
    label_encoder = LabelEncoder()
    from keras.models import model_from_json
    x_test, y_test = prepare_image_test()
    y_test = label_encoder.fit_transform(y_test)
    y_test = keras.utils.to_categorical(y_test, 4)
    # load json and create model
    json_file = open('model_cv/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_cv/model.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    test =loaded_model.predict(x_test, batch_size= 64, verbose=1)
    # print(test)
    score = loaded_model.evaluate(x_test, y_test, verbose=1)
    return(loaded_model.metrics_names[1], score[1]*100)



# if __name__ == "__main__":
#     train_model()

