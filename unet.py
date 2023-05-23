import os
import skimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from keras.layers import *
from keras import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


def build_model():
    inputs = Input(shape=(256, 256, 3))
    conv1 = Conv2D(32, (5, 5), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv2D(32, (5, 5), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (5, 5), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv2D(64, (5, 5), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (5, 5), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv2D(128, (5, 5), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (5, 5), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Conv2D(256, (5, 5), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (5, 5), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv2D(512, (5, 5), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    up6 = concatenate([Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (5, 5), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv2D(256, (5, 5), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (5, 5), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv2D(128, (5, 5), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    up8 = concatenate([Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (5, 5), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv8 = Conv2D(64, (5, 5), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    up9 = concatenate([Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (5, 5), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv9 = Conv2D(16, (5, 5), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)

    outputs = Conv2D(1, (1, 1), activation="relu")(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def train_model(path_to_train_data: str, path_to_checkpoint: str, model: Model):
    j2k_train_files = []
    tif_train_files = []

    filenames = os.listdir(path_to_train_data)
    for filename in filenames:
        if filename[-3:] == "j2k":
            j2k_train_files.append(filename)
        if filename[-3:] == "tif":
            tif_train_files.append(filename)

    j2k_train_files.sort()
    tif_train_files.sort()

    dim = (256, 256)
    X0 = []
    tifs0 = []
    num_images = len(tif_train_files)

    for i in range(num_images):
        img = np.array(skimage.io.imread(path_to_train_data + "/" + j2k_train_files[i]))
        mask = np.array(skimage.io.imread(path_to_train_data + "/" + tif_train_files[i]))
        resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resize_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
        X0.append(resize_img)
        tifs0.append(resize_mask)

    X0 = np.array(X0)
    tifs0 = np.array(tifs0)

    # Threshold
    threshold_our = 3000
    max_num = 100

    for i in range(tifs0.shape[0]):
        # Число пикселей со значениями, превышающими значение порога, меньше max_num
        # и хотя бы один пиксель не NaN
        if (np.sum(tifs0[i] >= threshold_our) < max_num) and np.max(tifs0[i] < 65535):
            # В пиксели, значения которых превышают значения порога, записывается медианное значение
            tifs0[i][(tifs0[i] >= threshold_our) & (tifs0[i] < 65535)] = np.median(tifs0[i])

    # Sanity check
    threshold = 10000

    # То же самое, но порог 10000 и макс. число пикселей - 50.
    for i in range(tifs0.shape[0]):
        if (np.sum(tifs0[i] >= threshold) < 50) and np.max(tifs0[i] < 65535):
            tifs0[i][(tifs0[i] >= threshold) & (tifs0[i] < 65535)] = np.median(tifs0[i])

    # Исключить из тестовой выборки изображения, содержащие NaN
    idx_keep = np.argwhere(np.sum(np.sum(tifs0 == 65535, axis=1), axis=1) == 0).reshape(-1)
    tifs0 = tifs0[idx_keep]
    X0 = X0[idx_keep]

    # Find the median pixel value for each image
    tif_medians = np.median(tifs0, axis=(1, 2))

    # Find the number of tifs with median of either min or max value
    # These could be examples of bad data
    count = 0
    for i in range(tif_medians.shape[0]):
        min_value = np.min(tifs0[i])
        max_value = np.max(tifs0[i])
        if (tif_medians[i] == min_value) or (tif_medians[i] == max_value):
            count += 1
    print(count)
    print(tifs0.shape)

    # Splitting elevation pixels into two classes based on if the median value is equal to
    # either the minimum or maximum value in the image. If the tif is kept, then it is
    # turned into a binary coloring based on the value of the median.
    for i in range(tif_medians.shape[0]):
        if tif_medians[i] == np.min(tifs0[i]):
            tifs0[i] = (tifs0[i] != np.min(tifs0[i]))
        elif tif_medians[i] == np.max(tifs0[i]):
            tifs0[i] = (tifs0[i] == np.max(tifs0[i]))
        else:
            tifs0[i] = tifs0[i] >= tif_medians[i]

    tifs0 = tifs0.astype("uint8")
    print(tifs0.shape)

    # Create the train and validation sets
    X_t, X_v, y_t, y_v = train_test_split(X0, tifs0, test_size=0.15, random_state=0)
    # Train the model
    path_to_checkpoint = path_to_checkpoint + 'best_model.h5'
    checkpoint = ModelCheckpoint(path_to_checkpoint, monitor='val_loss', save_best_only=True)
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=100, callbacks=[checkpoint])


def predict(model: Model, path_to_rgb: str):
    filenames_rgb = os.listdir(path_to_rgb)
    dim = (256, 256)
    X = []
    n = len(filenames_rgb)
    for i in range(n):
        img = np.array(skimage.io.imread(path_to_rgb + "/" + filenames_rgb[i]))
        resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        X.append(resize_img)

    X = np.array(X)
    tifs_test = model.predict(X).reshape(5, 256, 256)
    return tifs_test


def save_predictions(tifs, path_to_save):
    for i in range(len(tifs)):
        save_img(tifs[i], path_to_save)


def get_max_height(my_image):
    k = sorted(set(np.array(my_image).flatten()))
    k = np.unique(k)
    k.sort()
    if k[-1] == 65535:
        m = int(k[-2]) + 1
    else:
        m = int(k[-1]) + 1
    return m


def save_img(img, path_to_save, name="pred"):
    fig = plt.figure(figsize=(333, 333), dpi=1)
    m0 = get_max_height(img)

    plt.imshow(img, vmax=m0)
    plt.axis('off')

    plt.savefig(path_to_save + name + '_visual.tif', dpi=1, bbox_inches='tight', pad_inches=0)

    # plt.show()
