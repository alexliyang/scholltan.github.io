from __future__ import print_function

from keras import backend as K
from keras import layers
from keras.layers import Dropout, GlobalMaxPooling2D, Flatten, Activation, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Lambda,Dense
from keras.layers.convolutional import SeparableConv2D
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.topology import Layer
import tensorflow as tf


K.set_image_data_format('channels_last')  # TF dimension ordering in this code


import numpy as np
import warnings
from keras.utils import layer_utils
from keras.utils.data_utils import get_file



def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x




def loss_DSSIM(y_true, y_pred):
    # y_true = inputs[0]
    # y_pred = inputs[1]
    # print(y_true.shape)

    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    eps = 1e-9
    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
        
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
        
    covar_true_pred = K.mean(patches_true * patches_pred, axis=3) - u_true * u_pred
        
    std_true = K.sqrt(var_true+eps)
    std_pred = K.sqrt(var_pred+eps)
        
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (K.sqrt(u_true+eps) + K.sqrt(u_pred+eps) + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return K.mean(((1.0 - ssim) / 2))


class Selection(Layer):
    def __init__(self, disparity_levels=None, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        if disparity_levels is None:
            disparity_levels = range(0, 64, 2)

        super(Selection, self).__init__(**kwargs)

        self.disparity_levels = disparity_levels

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Selection` layer should be called '
                             'on a list of 2 inputs.')

    def call(self, inputs):

        # first we extract the left image from the original input
        image = inputs[0]
        # then the calculated disparity map that is the ouput of the Unet
        disparity_map = inputs[1]
        # initialize the stack of shifted left images
        shifted_images = []
        # loop over the different disparity levels and shift the left image accordingly, add it to the list
        for shift in self.disparity_levels:
            if shift > 0:
                shifted_images += [K.concatenate([image[..., shift:, :], K.zeros_like(image[..., :shift, :])], axis=2)]
            elif shift < 0:
                shifted_images += [K.concatenate([K.zeros_like(image[..., shift:, :]), image[..., :shift, :]], axis=2)]
            else:
                shifted_images += [image]

        # create a tensor of shape (None, im_rows, im_cols, disparity_levels)
        shifted_images_stack = K.stack(shifted_images)
        shifted_images_stack = K.permute_dimensions(shifted_images_stack, (1, 2, 3, 0, 4))

        # take the dot product with the disparity map along the disparity axis
        # and output the resulting right image of size (None, im_rows, im_cols)
        new_image = []
        for ch in range(3):
            new_image += [K.sum(shifted_images_stack[..., ch] * disparity_map, axis=3)]

        new_image = K.stack(new_image)
        new_image = K.permute_dimensions(new_image, (1, 2, 3, 0))

        return new_image

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Gradient(Layer):
    def __init__(self, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        super(Gradient, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, inputs):
        dinputs_dx_0 = inputs - K.concatenate([K.zeros_like(inputs[..., :1, :]), inputs[..., :-1, :]], axis=1)
        dinputs_dx_1 = inputs - K.concatenate([inputs[..., 1:, :], K.zeros_like(inputs[..., :1, :])], axis=1)

        dinputs_dy_0 = inputs - K.concatenate([K.zeros_like(inputs[..., :1]), inputs[..., :-1]], axis=2)
        dinputs_dy_1 = inputs - K.concatenate([inputs[..., 1:], K.zeros_like(inputs[..., :1])], axis=2)

        abs_gradient_sum = 0.25 * K.sqrt(
            K.square(dinputs_dx_0) + K.square(dinputs_dx_1) + K.square(dinputs_dy_0) + K.square(dinputs_dy_1))

        return abs_gradient_sum[..., 2:-2, 2:-2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 4, input_shape[2] - 4)


class Depth(Layer):
    def __init__(self, disparity_levels=None, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        if disparity_levels is None:
            disparity_levels = range(0, 64, 2)

        # if none, initialize the disparity levels as described in deep3d
        super(Depth, self).__init__(**kwargs)

        self.disparity_levels = disparity_levels

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, disparity):

        depth = []
        for n, disp in enumerate(self.disparity_levels):
            depth += [disparity[..., n] * disp]

        depth = K.concatenate(depth, axis=0)
        return K.sum(depth, axis=0, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def get_unet(img_rows, img_cols, lr=1e-4):
    inputs = Input((img_rows, 2 * img_cols, 3))  # 2 channels: left and right images

    # split input left/right wise
    left_input_image = Lambda(lambda x: x[..., :img_cols, :])(inputs)
    right_input_image = Lambda(lambda x: x[..., img_cols:, :])(inputs)

    concatenated_images = concatenate([left_input_image, right_input_image], axis=3)

    # x = ZeroPadding2D((3, 3))(concatenated_images)
    x1 = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(concatenated_images)
    x1 = BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((3, 3)(x1)

    x2 = conv_block(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x2 = identity_block(x2, 3, [64, 64, 256], stage=2, block='b')
    x2 = identity_block(x2, 3, [64, 64, 256], stage=2, block='c')

    x3 = conv_block(x2, 3, [128, 128, 512], stage=3, block='a')
    x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='b')
    x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='c')
    x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='d')

    x4 = conv_block(x3, 3, [256, 256, 1024], stage=4, block='a')
    x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='b')
    x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='c')
    x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='d')
    x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='e')
    x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='f')

    x5 = conv_block(x4, 3, [512, 512, 2048], stage=5, block='a')
    x5 = identity_block(x5, 3, [512, 512, 2048], stage=5, block='b')
    x5 = identity_block(x5, 3, [512, 512, 2048], stage=5, block='c')

    x6 = concatenate([Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(x5), x4], axis=3)
    x6 = conv_block(x6, 3, [256, 256, 1024], stage=6, block='a', strides=(1, 1))
    x6 = identity_block(x6, 3, [256, 256, 1024], stage=6, block='b')
    x6 = identity_block(x6, 3, [256, 256, 1024], stage=6, block='c')
    x6 = identity_block(x6, 3, [256, 256, 1024], stage=6, block='d')
    x6 = identity_block(x6, 3, [256, 256, 1024], stage=6, block='e')
    x6 = identity_block(x6, 3, [256, 256, 1024], stage=6, block='f')
    # x6 = Dropout(rate=0.5)(x6)

    x7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x6), x3], axis=3)
    x7 = conv_block(x7, 3, [128, 128, 512], stage=7, block='a', strides=(1, 1))
    x7 = identity_block(x3, 3, [128, 128, 512], stage=7, block='b')
    x7 = identity_block(x3, 3, [128, 128, 512], stage=7, block='c')
    x7 = identity_block(x3, 3, [128, 128, 512], stage=7, block='d')
    # x7 = Dropout(rate=0.5)(x7)

    x8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x7), x2], axis=3)
    x8 = conv_block(x8, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x8 = identity_block(x8, 3, [64, 64, 256], stage=2, block='b')
    x8 = identity_block(x8, 3, [64, 64, 256], stage=2, block='c')

    x9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x8), x1], axis=3)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same')(x9)
    x9 = BatchNormalization()(x9)
    x9 = SeparableConv2D(64, (3, 3), activation='relu', padding='same'(x9)
    x9 = BatchNormalization()(x9)
    # x9 = Dropout(rate=0.4)(x9)

    # split into left/right disparity maps

    left_disparity_level_4 = Conv2DTranspose(32, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., 512:])(x4))
    right_disparity_level_4 = Conv2DTranspose(32, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., :512])(x4))

    left_disparity_level_3 = Conv2DTranspose(32, (8, 8), strides=(8, 8), padding='same')(
        Lambda(lambda x: x[..., 256:])(x3))
    right_disparity_level_3 = Conv2DTranspose(32, (8, 8), strides=(8, 8), padding='same')(
        Lambda(lambda x: x[..., :256])(x3))

    left_disparity_level_2 = Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., 128:])(x2))
    right_disparity_level_2 = Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., :128])(x2))

    left_disparity_level_1 = Lambda(lambda x: x[..., :32])(x9)
    right_disparity_level_1 = Lambda(lambda x: x[..., 32:])(x9)

    left_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([left_disparity_level_1,
                                                                                  left_disparity_level_2,
                                                                                  left_disparity_level_3,
                                                                                  left_disparity_level_4])

    right_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([right_disparity_level_1,
                                                                                   right_disparity_level_2,
                                                                                   right_disparity_level_3,
                                                                                   right_disparity_level_4])

    # use a softmax activation on the conv layer output to get a probabilistic disparity map
    left_disparity = SeparableConv2D(32, (3, 3), activation='softmax', padding='same')(left_disparity)

    right_disparity = SeparableConv2D(32, (3, 3), activation='softmax', padding='same')(right_disparity)

    left_disparity_levels = range(0, 64, 2)
    right_reconstruct_im = Selection(disparity_levels=left_disparity_levels)([left_input_image, left_disparity])

    right_disparity_levels = range(-64, 0, 2)
    left_reconstruct_im = Selection(disparity_levels=right_disparity_levels)([right_input_image, right_disparity])

    # concatenate left and right images along the channel axis
    output = concatenate([left_reconstruct_im, right_reconstruct_im], axis=2, name = 'output')

    # gradient regularization:
    depth_left = Depth(disparity_levels=left_disparity_levels)(left_disparity)
    depth_right = Depth(disparity_levels=left_disparity_levels)(right_disparity)
    depth_left_gradient = Gradient()(depth_left)
    depth_right_gradient = Gradient()(depth_right)

    left_input_im_gray = Lambda(lambda x: K.mean(x, axis=3))(left_input_image)
    right_input_im_gray = Lambda(lambda x: K.mean(x, axis=3))(right_input_image)

    left_input_im_gray_norm = Lambda(lambda x: x / K.max(x))(left_input_im_gray)
    right_input_im_gray_norm = Lambda(lambda x: x / K.max(x))(right_input_im_gray)

    image_left_gradient = Gradient()(left_input_im_gray_norm)
    image_right_gradient = Gradient()(right_input_im_gray_norm)

    weighted_gradient_left = Lambda(lambda x: x[0] * (1 - x[1]), name='weighted_gradient_left')([depth_left_gradient, image_left_gradient])
    weighted_gradient_right = Lambda(lambda x: x[0] * (1 - x[1]), name='weighted_gradient_right')([depth_right_gradient, image_right_gradient])

    #  add a ssim layer
    loss_ssim_recon = concatenate([left_reconstruct_im, right_reconstruct_im], axis=2, name = 'loss_ssim_recon')

    model = Model(inputs=[inputs], outputs=[output, loss_ssim_recon, weighted_gradient_left, weighted_gradient_right])

    disp_map_model = Model(inputs=[inputs], outputs=[left_disparity, right_disparity])


    # model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error', loss_weights=[1., 0.02, 0.001, 0.001])
    # model.summary()
    model.compile(optimizer=Adam(lr=lr),
              loss={'output': 'mean_absolute_error', 'loss_ssim_recon': loss_DSSIM, 
              'weighted_gradient_left':'mean_absolute_error', 'weighted_gradient_right':'mean_absolute_error'},
              loss_weights={'output': 1., 'loss_ssim_recon': 0.02, 'weighted_gradient_left':0.001,'weighted_gradient_right':0.001})
    model.summary()

    return model, disp_map_model
