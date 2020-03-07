import keras
import cv2
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


def img_model():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
                # ESC pressed
            print("Escape hit, closing...")
            break
        # else :
        elif k%256 == 32:
            path = "dataset/test_data/opencv_frame_0.png"
            # cv2.imwrite(path, frame)
            x = []
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            x.append(img_array)
            x = np.array(x)
            x = keras.applications.mobilenet.preprocess_input(x)
            from keras.models import model_from_json
            # load json and create model
            json_file = open('model_cv\model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model_cv\model.h5")
            print("Loaded model from disk")

            # evaluate loaded model on test data
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            score = loaded_model.predict(x, batch_size= 64, verbose=1)

            
            X_train = [ [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]
            
            y_train = [1, 2, 3, 4]
            clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(score)
            print(y_pred)

    cam.release()

    cv2.destroyAllWindows()
# if __name__ == "__main__":
#     img_model()


