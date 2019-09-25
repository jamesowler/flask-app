from keras.optimizers import Adam
from keras.applications import ResNet152
from keras.applications.resnet50 import decode_predictions
from keras_efficientnets import EfficientNetB4
import cv2
import matplotlib.pyplot as plt
import numpy as np

from classifier.classes import classes


def efficient_net():

    '''
    Constructed by scaling up CNN in more structure mannner
    '''

    model = EfficientNetB4(input_shape=(380, 380, 3), classes=1000, include_top=True, weights='imagenet')
    return model

def resnet():
    model = ResNet152(weights='imagenet', include_top=True)
    return model


def predict_class(file_path, model, input_size = (224, 224)):
    img = cv2.imread(file_path)
    # convert to rgb
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize to fit input size of model
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    # batch of 1
    img = np.expand_dims(img, axis=0)

    y_pred = model.predict(img)
    y_class = decode_predictions(y_pred, top=1)

    return img, y_pred, y_class


if __name__ == '__main__':

    model = resnet()
    img, y_pred = predict_class('/home/james/Downloads/IMG_0486.jpg', model)
    print('Predicted: ', decode_predictions(y_pred, top=3)[0])
