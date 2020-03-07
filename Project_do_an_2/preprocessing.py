from created_dataset import classification
import keras
import cv2
import numpy as np
from keras.applications.mobilenet import preprocess_input 
from keras.preprocessing import image 

def take_photos():
    classification("class_tien")
    classification("class_lui")
    classification("class_trai")
    classification("class_phai")

# mobile = keras.applications.mobilenet.MobileNet()
def prepare_image_train():
    # take_photos()
    label = ["class_tien", "class_lui", "class_trai", "class_phai"]
    k = 0 
    x_train = []
    y_train = []



    for i in label :
        k = k+1
        for photo in range(20):
            path = "dataset/image_{}/opencv_frame_{}.png".format(i, photo)
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            x_train.append(img_array)
            y_train.append(k)

   
    x_train = np.array(x_train)


    # x_train = np.expand_dims(x_train, axis = 0)
    # print(len(x_train))
    y_train = np.array(y_train)

    # print(y_train.shape)
    # y_train = np.expand_dims(y_train, axis = 1)
    

    return keras.applications.mobilenet.preprocess_input(x_train), y_train
def prepare_image_val():
    # take_photos()
    label = ["class_tien", "class_lui", "class_trai", "class_phai"]
    k = 0 
    x_train = []
    y_train = []



    for i in label :
        k = k+1
        for photo in range(20,25):
            path = "dataset/image_{}/opencv_frame_{}.png".format(i, photo)
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            x_train.append(img_array)
            y_train.append(k)

   
    x_train = np.array(x_train)


    # x_train = np.expand_dims(x_train, axis = 0)
    # print(len(x_train))
    y_train = np.array(y_train)

    # print(y_train.shape)
    # y_train = np.expand_dims(y_train, axis = 1)

    return keras.applications.mobilenet.preprocess_input(x_train), y_train
def prepare_image_test():
    # take_photos()
    label = ["class_tien", "class_lui", "class_trai", "class_phai"]
    k = 0 
    x_train = []
    y_train = []



    for i in label :
        k = k+1
        for photo in range(25,30):
            path = "dataset/image_{}/opencv_frame_{}.png".format(i, photo)
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            x_train.append(img_array)
            y_train.append(k)

   
    x_train = np.array(x_train)


    # x_train = np.expand_dims(x_train, axis = 0)
    # print(len(x_train))
    y_train = np.array(y_train)

    # print(y_train.shape)
    # y_train = np.expand_dims(y_train, axis = 1)

    return keras.applications.mobilenet.preprocess_input(x_train), y_train






