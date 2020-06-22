
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Dense, concatenate,Dropout, SpatialDropout2D,Activation,Add,multiply
from keras.layers.merge import add as keras_add
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K
from subtle_metrics import mixedLoss
from keras.models import model_from_yaml, model_from_json
from keras.layers.core import Activation, Layer
from keras.regularizers import l2,l1

import numpy as np

# clean up
def clearKerasMemory():
    K.clear_session()

# use part of memory
def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))

# load models
def loadKerasModel(filepath, string_model=None, format_export='json'):
    if string_model is None:
        with open(filepath,'r') as file:
            string_model = file.read()
    if format_export == 'json':
        model = model_from_json(string_model)
    else:
        model = model_from_yaml(string_model)
    return model

def create_attention_block_2D(g, x, output_channel, padding='same'):
    g1 = Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(g)
    g1 = BatchNormalization(axis=-1)(g1)
    x1 = Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    psi = Activation("relu")(Add()([g1, x1]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding=padding)(psi)
    psi = BatchNormalization(axis=-1)(psi)
    psi = Activation("sigmoid")(psi)
    return multiply([x, psi])



def dropoutResUNet(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128,
                   y=np.array([-1, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function=mixedLoss(),
                   metrics_monitor=[mean_absolute_error, mean_squared_error],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,
                   with_bn=True,  # don't use for F16 now
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            x = BatchNormalization()(x)
            return Activation(activation_conv)(x)
    else:
        def lambda_bn(x):
            return x

    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # step1
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        conv1 = lambda_bn(conv1)
        conv1 = Conv2D(num_channel_first, (3, 3),
                       padding="same",
                       activation=activation_conv,
                       kernel_initializer=kernel_initializer)(conv1)

        if (i + 1) % 2 == 0 and i != 1:
            conv1 = keras_add([conv_identity[-1], conv1])
            pdb.set_trace() # jiahong apr
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            conv_encoder = lambda_bn(conv_encoder)
            conv_encoder = Conv2D(
                num_channel, (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_encoder = keras_add([conv_identity[-1], conv_encoder])
                pdb.set_trace() # jiahong apr
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros')(pool_encoders[-1])
    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros')(conv_center)


    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):

        attention_gated = create_attention_block_2D(Conv2DTranspose(
            list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
            activation=activation_conv,
            kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i],list_num_features[-i])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), attention_gated])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            conv_decoder = lambda_bn(conv_decoder)
            conv_decoder = Conv2D(
                list_num_features[-i], (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
                pdb.set_trace() # jiahong apr
        conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])
        pdb.set_trace() # jiahong apr

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model
