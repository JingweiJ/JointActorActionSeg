import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
from keras.layers import TimeDistributed as TD
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import numpy as np
import sys
sys.path.append('..')
from models.batchnorm import BatchNorm
from pdb import set_trace

def identity_block_3d(input_tensor, s_kernel_size, t_kernel_size, filters, stage,
                      block, training, use_bias=True, temporal=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor # TODO: shape?
        s_kernel_size: defualt 3, the kernel size of middle spatial conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    spatial_conv_name_base = 'res' + str(stage) + block + '_branch'
    temporal_conv_name_base = 'tres' + str(stage) + block + '_branch'
    spatial_bn_name_base = 'bn' + str(stage) + block + '_branch'
    temporal_bn_name_base = 'tbn' + str(stage) + block + '_branch'

    # 1x1x1
    x = KL.TimeDistributed(KL.Conv2D(nb_filter1, (1,1), use_bias=use_bias),
                           name=spatial_conv_name_base + '2a')(input_tensor)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    # 1xkxk
    x = KL.TimeDistributed(KL.Conv2D(nb_filter2, (s_kernel_size, s_kernel_size),
                                     padding='same'), name=spatial_conv_name_base + '2b')(x)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    if temporal:
        # kx1x1 # TODO: add back
        x = KL.Conv3D(nb_filter2, (t_kernel_size,1,1), padding='same',
                      name=temporal_conv_name_base + '2b')(x)
        #x = KL.BatchNormalization(axis=4, name=temporal_bn_name_base + '2b')(x, training=training)
        x = KL.Activation('relu')(x)

    # 1x1x1
    x = KL.TimeDistributed(KL.Conv2D(nb_filter3, (1,1), use_bias=use_bias),
                           name=spatial_conv_name_base + '2c')(x)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2c')(x)

    # Add shortcut
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x

def conv_block_3d(input_tensor, s_kernel_size, t_kernel_size, filters, stage,
               block, training, strides=(2,2), use_bias=True, temporal=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        s_kernel_size: defualt 3, the kernel size of middle spatial conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    spatial_conv_name_base = 'res' + str(stage) + block + '_branch'
    temporal_conv_name_base = 'tres' + str(stage) + block + '_branch'
    spatial_bn_name_base = 'bn' + str(stage) + block + '_branch'
    temporal_bn_name_base = 'tbn' + str(stage) + block + '_branch'

    # 1x1x1
    x = KL.TimeDistributed(KL.Conv2D(nb_filter1, (1,1), strides=strides, use_bias=use_bias),
                           name=spatial_conv_name_base + '2a')(input_tensor)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    # 1xkxk
    x = KL.TimeDistributed(KL.Conv2D(nb_filter2, (s_kernel_size, s_kernel_size),
                                     padding='same'), name=spatial_conv_name_base + '2b')(x)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    if temporal:
        # kx1x1 # TODO: add back
        x = KL.Conv3D(nb_filter2, (t_kernel_size,1,1),
                      padding='same', name=temporal_conv_name_base + '2b')(x)
        #x = KL.BatchNormalization(axis=4, name=temporal_bn_name_base + '2b')(x, training=training)
        x = KL.Activation('relu')(x)

    # 1x1x1
    x = KL.TimeDistributed(KL.Conv2D(nb_filter3, (1,1), use_bias=use_bias),
                           name=spatial_conv_name_base + '2c')(x)
    x = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '2c')(x)

    # Add shortcut
    shortcut = KL.TimeDistributed(KL.Conv2D(nb_filter3, (1,1), strides=strides, use_bias=use_bias),
                                  name=spatial_conv_name_base + '1')(input_tensor)
    shortcut = TD(BatchNorm(axis=3), name=spatial_bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x

def resnet_3d_graph(input_clip, mode, architecture, stage5=False, temporal=True):
    # input_clip: [bs, timesteps, height, width, channel]
    # TODO: add a param in config indicating which stages contain temporal layers
    assert mode in ['training', 'inference']
    training = True if mode == 'training' else False
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding3D((0, 3, 3))(input_clip)
    x = KL.TimeDistributed(KL.Conv2D(64, (7, 7), strides=(2, 2), use_bias=True), name='conv1')(x)
    x = TD(BatchNorm(axis=3), name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.TimeDistributed(KL.MaxPooling2D((3,3), strides=(2,2), padding='same'))(x)
    # Stage 2
    x = conv_block_3d(x, 3, 3, [64, 64, 256], stage=2, block='a', training=training, strides=(1, 1))
    x = identity_block_3d(x, 3, 3, [64, 64, 256], stage=2, block='b', training=training)
    C2 = x = identity_block_3d(x, 3, 3, [64, 64, 256], stage=2, block='c', training=training)
    # Stage 3
    x = conv_block_3d(x, 3, 3, [128, 128, 512], stage=3, block='a', training=training)
    x = identity_block_3d(x, 3, 3, [128, 128, 512], stage=3, block='b', training=training)
    x = identity_block_3d(x, 3, 3, [128, 128, 512], stage=3, block='c', training=training)
    C3 = x = identity_block_3d(x, 3, 3, [128, 128, 512], stage=3, block='d', training=training)
    # Stage 4
    x = conv_block_3d(x, 3, 3, [256, 256, 1024], stage=4, block='a', training=training)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block_3d(x, 3, 3, [256, 256, 1024], stage=4, block=chr(98+i), training=training, temporal=temporal)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block_3d(x, 3, 3, [512, 512, 2048], stage=5, block='a', training=training, temporal=temporal)
        x = identity_block_3d(x, 3, 3, [512, 512, 2048], stage=5, block='b', training=training, temporal=temporal)
        C5 = x = identity_block_3d(x, 3, 3, [512, 512, 2048], stage=5, block='c', training=training, temporal=temporal)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

if __name__ == '__main__':
    input_clip = KL.Input(shape=(16, 256, 256, 3))
    C1, C2, C3, C4, C5 = resnet_3d_graph(input_clip, 'resnet50', stage5=True)
    model = KM.Model([input_clip], [C1, C2, C3, C4, C5])
    set_trace()
    #from keras.utils import plot_model
    #plot_model(model, './resnet_3d.png', show_shapes=True)
