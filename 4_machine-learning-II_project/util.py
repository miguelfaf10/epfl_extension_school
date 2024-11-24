import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def get_image_by_index(data_flow, desired_index):

    image = np.array(Image.open(data_flow.filepaths[desired_index]))
    label = data_flow.labels[desired_index]

    return image, label


def decode_class(y):
    return np.argmax(y,axis=1)