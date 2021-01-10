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

def get_unet(input_img, n_filters=128, dropout=0.15, batchnorm=True, bands=11):
  d1=DownBlock(input_img,bands,128,3, True,12)
  d2=DownBlock(d1,128,256)
  d3=DownBlock(d2,256,512)
  d4=DownBlock(d3,512,1024)
  u1=UpBlock(d4,d3,1024,512)
  u2=UpBlock(u1,d2,512,256,3,True,12)
  u3=UpBlock(u2,d1,256,128)
  
  outputs = Conv2DTranspose(filters=2, kernel_size=(3,3),strides=(2,2),padding="same", kernel_initializer="he_normal")(u3)
  outputs = core.Reshape((128*128,2))(outputs)
  outputs = core.Activation('softmax')(outputs)
      
  model = Model(inputs=[input_img], outputs=[outputs])
  return model