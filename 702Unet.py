import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
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

from random import shuffle

im_width = 128
im_height = 128
border = 5
inputFolder = 'input/data1'
targetFolder = 'target/'

ids = next(os.walk(targetFolder))[2]
print(len(ids))
shuffle(ids)

train_ids = ids[0:int(len(ids) * 0.8)]
valid_ids = ids[int(len(ids) * 0.8):]

print(train_ids)
print(valid_ids)
print(len(train_ids))
print(len(valid_ids))

def conv2d_block(input_tensor, n_filters, kernel_size=3,strides=1,batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),strides=(strides, strides),padding="same", kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def encoder_decoder(x1,ni, kernel_size=3, batchnorm=True,times=None):
  x=GlobalMaxPooling2D()(x1)
  x=Reshape(target_shape=(1,1,times))(x)
  x=Conv2D(filters=ni//2, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x)
  x=LeakyReLU(alpha=0.1)(x)
  x=Conv2D(filters=ni, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x)
  x=Activation('sigmoid')(x)
  
  x2=Conv2D(filters=ni, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x1)
  x2=Activation('sigmoid')(x2)
  
  
  x11=Multiply()([x1,x2])
  x12=Multiply()([x1,x])
  x13=Add()([x11,x12])
  
  return x13

def DownBlock(x,ni,nf, kernel_size=3, batchnorm=True,down=None):
  inp=x
  x=conv2d_block(x,nf,3,2)
  x=conv2d_block(x,nf,3)
  x=Add()([x,conv2d_block(inp,nf,3,2)])
  if down is not None:
    return encoder_decoder(x,nf, kernel_size=3, batchnorm=True,times=128)
  else:
    return x

def UpBlock(down,cross,ni,nf, kernel_size=3, batchnorm=True,down1=None):
  x=Conv2DTranspose(filters=nf, kernel_size=(3, 3),strides=(2,2),padding="same", kernel_initializer="he_normal")(down)
  print(x)
  print(cross)
  x=concatenate([x,cross])
  x=conv2d_block(x,nf,3)
  if down1 is not None:
    return encoder_decoder(x,nf, kernel_size=3, batchnorm=True,times=256)
  else:
    return x

def generateUnet(input_img, n_filters=128, dropout=0.15, batchnorm=True):
    d1=DownBlock(input_img,7,128,3, True,12)
    d2=DownBlock(d1,128,256)
    d3=DownBlock(d2,256,512)
    d4=DownBlock(d3,512,1024)
    u1=UpBlock(d4,d3,1024,512)
    u2=UpBlock(u1,d2,512,256,3,True,12)
    u3=UpBlock(u2,d1,256,128)
    outputs = Conv2DTranspose(filters=255, kernel_size=(3,3),strides=(2,2),padding="same", kernel_initializer="he_normal")(u3)

    outputs = core.Reshape((128*128,255))(outputs)
    outputs = core.Activation('softmax')(outputs)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((128, 128, 12), name='img')
model = generateUnet(input_img, n_filters=7, dropout=0.15, batchnorm=True)
model.compile(optimizer=Adam(lr=0.1), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor='loss',patience=10, verbose=1),
    ReduceLROnPlateau(monitor='loss',factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
    ModelCheckpoint('model3.h5', monitor='loss',verbose=1, save_best_only=True, save_weights_only=False)
]

xStep = 6
yStep = 4

def get_data(ids, batch_size):
    while True:
        ids_batches = [ids[i:min(i+batch_size,len(ids))] for i in range(0, len(ids), batch_size)] 
        for xC in range(xStep):
            for yC in range(yStep):
                k=-1
                X = np.zeros((len(ids_batches[b]), im_height, im_width, 12), dtype=np.float32)
                y = np.zeros((len(ids_batches[b]), im_height * im_width, 255), dtype=np.float32)
                for c in range(len(ids_batches[b])):
                    k=k+1

                for r in range(1,12):
                    imgfilename = "{}/T36TVK_20200829T083611_B{}_20m.jp2.tif_x_{}_y_{}.tif".format(inputFolder, r, xC, yC)
                    img = load_img(imgfilename, color_mode="grayscale")
                    x_img = img_to_array(img)
                    x_img = resize(x_img, (128, 128), mode='constant', preserve_range=True)
                    for p in range(128):
                        for q in range(128):
                            #print(x_img[p][q]/255)
                            X[k][p][q][r-1]=x_img[p][q]/255
                    
                #k=k+1
                # Save images
                #X[k, ..., 0] = temp1 / 255  

                # Load masks

                mask = img_to_array(load_img(path_train_output+ids_batches[b][c], color_mode="grayscale"))
                mask = resize(mask, (128, 128), mode='constant', preserve_range=True)

                inc=-1
                for p in range(128):
                    for q in range(128):
                        num=int(mask[p][q])
                        temp=np.zeros((255), dtype=np.float32)
                        temp[num]=1
                        inc=inc+1
                        y[k][inc]=temp

            yield X,y
            


train_generator = get_data(train_ids[0:1], batch_size=1)
valid_generator = get_data(valid_ids[0:1], batch_size=1)