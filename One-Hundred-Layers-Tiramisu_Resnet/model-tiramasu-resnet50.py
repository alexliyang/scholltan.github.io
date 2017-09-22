from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras import layers
from keras.layers import add

from keras.layers import Conv2D, Conv2DTranspose

from keras import backend as K



from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D


import cv2
import numpy as np
import json

K.set_image_dim_ordering('tf')

from keras.regularizers import l2
 
class Tiramisu():

    def __init__(self):
        self.create()

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        bn_axis = 3
        conv_name_base = 'res_id' + str(stage) + block + '_branch'
        bn_name_base = 'bn_id' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), padding='same', kernel_initializer="he_uniform", name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,padding='same', kernel_initializer="he_uniform", name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001), name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), padding='same', kernel_initializer="he_uniform", name=conv_name_base + '2c')(x)
        # x = Dropout(rate=0.2)(x)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001), name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides, padding='same', kernel_initializer="he_uniform", 
                    name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer="he_uniform", 
                    name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001), name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), padding='same', kernel_initializer="he_uniform", name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001), name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, padding='same', kernel_initializer="he_uniform", 
                            name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001), name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def create(self):
        inputs = Input((224,224,3))
        bn_axis = 3

        x0 = Conv2D(64, kernel_size=(7, 7), padding='same', strides=(1, 1),
                                kernel_initializer="he_uniform",
                                kernel_regularizer = l2(0.0001))(inputs)
        x0 = BatchNormalization(axis=bn_axis, name='bn_conv1',
                                gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001))(x0)

        x0 = Activation('relu')(x0)
        x1 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x0)
        print(x0.shape) 
        print(x1.shape)

        x2 = self.conv_block(x1, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
        x2 = self.identity_block(x2, 3, [64, 64, 128], stage=2, block='b')
        x2 = self.identity_block(x2, 3, [64, 64, 128], stage=2, block='c')
        print(x2.shape)

        x3 = self.conv_block(x2, 3, [128, 128, 256], stage=3, block='a')
        x3 = self.identity_block(x3, 3, [128, 128, 256], stage=3, block='b')
        x3 = self.identity_block(x3, 3, [128, 128, 256], stage=3, block='c')
        x3 = self.identity_block(x3, 3, [128, 128, 256], stage=3, block='d')
        print(x3.shape)

        x4 = self.conv_block(x3, 3, [256, 256, 512], stage=4, block='a')
        x4 = self.identity_block(x4, 3, [256, 256, 512], stage=4, block='b')
        x4 = self.identity_block(x4, 3, [256, 256, 512], stage=4, block='c')
        x4 = self.identity_block(x4, 3, [256, 256, 512], stage=4, block='d')
        x4 = self.identity_block(x4, 3, [256, 256, 512], stage=4, block='e')
        x4 = self.identity_block(x4, 3, [256, 256, 512], stage=4, block='f')
        print(x4.shape)

        x5 = self.conv_block(x4, 3, [512, 512, 1024], stage=5, block='a')
        x5 = self.identity_block(x5, 3, [512, 512, 1024], stage=5, block='b')
        x5 = self.identity_block(x5, 3, [512, 512, 1024], stage=5, block='c')
        print(x5.shape)

        x6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x5), x4], axis=3)
        x6 = self.conv_block(x6, 3, [256, 256, 512], stage=6, block='a', strides=(1, 1))
        x6 = self.identity_block(x6, 3, [256, 256, 512], stage=6, block='b')
        x6 = self.identity_block(x6, 3, [256, 256, 512], stage=6, block='c')
        x6 = self.identity_block(x6, 3, [256, 256, 512], stage=6, block='d')
        x6 = self.identity_block(x6, 3, [256, 256, 512], stage=6, block='e')
        x6 = self.identity_block(x6, 3, [256, 256, 512], stage=6, block='f')
    	# x6 = Dropout(rate=0.5)(x6)
        print(x6.shape)

        x7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x6), x3], axis=3)
        x7 = self.conv_block(x7, 3, [128, 128, 256], stage=7, block='a', strides=(1, 1))
        x7 = self.identity_block(x3, 3, [128, 128, 256], stage=7, block='b')
        x7 = self.identity_block(x3, 3, [128, 128, 256], stage=7, block='c')
        x7 = self.identity_block(x3, 3, [128, 128, 256], stage=7, block='d')
        # x7 = Dropout(rate=0.5)(x7)
        print(x7.shape)

        x8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x7), x2], axis=3)
        x8 = self.conv_block(x8, 3, [64, 64, 128], stage=8, block='a', strides=(1, 1))
        x8 = self.identity_block(x8, 3, [64, 64, 128], stage=8, block='b')
        x8 = self.identity_block(x8, 3, [64, 64, 128], stage=8, block='c')
        print(x8.shape)

        x9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x8), x0], axis=3)
        x9 = Conv2D(64, (3, 3), activation='relu', padding='same')(x9)
        x9 = BatchNormalization(axis=bn_axis, name='bn_conv9a')(x9)
        x9 = SeparableConv2D(64, (3,3), activation='relu', padding='same')(x9)
        x9 = BatchNormalization(axis=bn_axis, name='bn_conv9b')(x9)
        print(x9.shape)

        last_conv = Conv2D(12, activation='linear', kernel_size=(1,1), padding='same',
                                kernel_regularizer = l2(0.0001))(x9)
            
        reshape = Reshape((12, 224 * 224))(last_conv)
        perm = Permute((2, 1))(reshape)
        act = Activation('softmax')(perm)
        
        model = Model(inputs=[inputs], outputs=[act])

        with open('tiramisu_fc_resnet50_model_v1.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()
