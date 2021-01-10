import os
import terrainModel
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Lambda
from keras.layers import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import LeakyReLU

from keras.activations import softmax
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,GlobalMaxPooling2D
from keras.layers import Add,Multiply
from keras import backend as K
import math

xStep = 6
yStep = 4

im_width = 128
im_height = 128

def get_data(inpaths, lengths):
  while True:
    for path in inpaths:     
      for xC in range(xStep):
        for yC in range(yStep):
          X = np.zeros((1, im_height, im_width, len(bands)), dtype=np.float32)
          y = np.zeros((1, im_height * im_width, 2), dtype=np.float32)
          r = 0
          for b in bands:
            filename = path + b + ".jp2.tif_x_" + str(xC) + "_y_" + str(yC) + ".tif"
            img = load_img(filename, color_mode="grayscale")
            x_img = img_to_array(img)
            x_img = resize(x_img, (128, 128), mode='constant', preserve_range=True)
            for p in range(128):
              for q in range(128):
                X[0][p][q][r] = x_img[p][q]/65535.0
            r = r + 1

          maskFilename = target_path + "/ground_truth_buildings_clipped.tif_x_" + str(xC) + "_y_" + str(yC) + ".tif"
          mask_img = load_img(maskFilename, color_mode="grayscale")

          mask = img_to_array(mask_img)
          mask = resize(mask, (128, 128), mode='constant', preserve_range=True)

          inc=0
          for p in range(128):
            for q in range(128):
              num=int(math.ceil(mask[p][q]))
              temp=np.zeros((2), dtype=np.float32)
              temp[num]=1
              y[0][inc]=temp
              inc=inc+1
       
          yield X,y

def get_data_data(inpaths, lengths):
  for path in inpaths:
    k = 0
    
    X = np.zeros((xStep * yStep, im_height, im_width, len(bands)), dtype=np.float32)
    y = np.zeros((xStep * yStep, im_height * im_width, 2), dtype=np.float32)
    for xC in range(xStep):
      for yC in range(yStep):
        r = 0
        for b in bands:
          filename = path + b + ".jp2.tif_x_" + str(xC) + "_y_" + str(yC) + ".tif"
          img = load_img(filename, color_mode="grayscale")
          x_img = img_to_array(img)
          x_img = resize(x_img, (128, 128), mode='constant', preserve_range=True)
          for p in range(128):
            for q in range(128):
              X[k][p][q][r] = x_img[p][q]/65535.0
          r = r + 1

        maskFilename = target_path + "/ground_truth_buildings_clipped.tif_x_" + str(xC) + "_y_" + str(yC) + ".tif"
        mask_img = load_img(maskFilename, color_mode="grayscale")

        mask = img_to_array(mask_img)
        mask = resize(mask, (128, 128), mode='constant', preserve_range=True)

        inc=0
        for p in range(128):
          for q in range(128):
            num=int(math.ceil(mask[p][q]))
            temp=np.zeros((2), dtype=np.float32)
            temp[num]=1
            y[k][inc]=temp
            inc=inc+1
        k = k + 1

    return X,y            
	
bands = ["B01_60m", "B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B09_60m", "B11_20m", "B12_20m"]
#bands = ["B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m"]


input_img = Input((128, 128, len(bands)), name='img')
model = terrainModel.get_unet(input_img, n_filters=7, dropout=0.15, batchnorm=True, bands=len(bands))

print(model)

model.compile(optimizer=Adam(lr=0.1), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

callbacks = [
    EarlyStopping(monitor='loss',patience=10, verbose=1),
    ReduceLROnPlateau(monitor='loss',factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
    ModelCheckpoint('model.h5', monitor='loss',verbose=1, save_best_only=True, save_weights_only=False)
]

                      
input_paths = ["input/data1/T36TVK_20200829T083611_", "input/data2/T36TVK_20200928T083741_"]
valid_paths = ["input/data3/T36TVK_20201028T084051_"]
target_path = "target"

ids1 = next(os.walk("input/data1"))[2];
ids2 = next(os.walk("input/data2"))[2];
ids3 = next(os.walk("input/data3"))[2];

totalLength = [ids1]
validL = [ids3]

train_generator = get_data(input_paths, totalLength)
valid_generator = get_data(valid_paths, validL)

results = model.fit(train_generator, steps_per_epoch=10 , epochs=500, verbose=1, validation_data=valid_generator, validation_steps=1, callbacks=callbacks)