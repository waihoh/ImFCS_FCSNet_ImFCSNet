import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv3D, BatchNormalization, Activation, Lambda, Dense, Input, Cropping1D, add
from tensorflow.keras.models import Model
from Configurations import CNN
from Utilities.common_packages import np

'''
Keras part: https://stats.stackexchange.com/questions/308795/standard-deviation-in-neural-network-regression
'''


def _BN_RELU(this_input, this_momentum, mdlname):
    layer = BatchNormalization(momentum=this_momentum, name=(mdlname + "_BN_RELU"))(this_input)
    return Activation(activation=CNN.ACTIVATION, name=(mdlname + "_Activation"))(layer)


def _CONV1D_BN_RELU(this_input, filters, temporal_filter_size, mdlname):
    layer = Conv1D(filters=filters, kernel_initializer=CNN.WT_INITIALIZER,
                   kernel_size=(temporal_filter_size,),
                   name=(mdlname + "_Conv1D"))(this_input)
    return _BN_RELU(this_input=layer, this_momentum=CNN.MOMENTUM, mdlname=mdlname)


def _shortcut(this_input, residual, mdlname):
    # truncate the end section of input
    shortcut = Cropping1D(cropping=(0, 2*CNN.TEMPORAL_FILTER_SIZE - 2), name=(mdlname + "_Cropping1D"))(this_input)
    return add([shortcut, residual])  # combine the two layers by adding.


def gen_model(outputdim, this_input, modelnum):
    # name of sub model
    mdlname = "mdl" + str(modelnum)

    # default CNN settings
    this_num_filters = CNN.NUM_FILTERS
    this_temporal_filter_size = CNN.TEMPORAL_FILTER_SIZE
    this_depth_1D = CNN.DEPTH_1D

    layer = Conv3D(this_num_filters, kernel_initializer=CNN.WT_INITIALIZER,
                   kernel_size=(CNN.FILTER_3D_SIZE, CNN.SPATIAL_FILTER_SIZE, CNN.SPATIAL_FILTER_SIZE), 
                   name=(mdlname + "_Conv3D"))(this_input)
    # remove spatial dimension. Done twice (i.e. for x & y dimensions).
    layer = Lambda(lambda arg: K.mean(arg, axis=-2), name=(mdlname + "_Lambda1"))(layer)
    layer = Lambda(lambda arg: K.mean(arg, axis=-2), name=(mdlname + "_Lambda2"))(layer)

    layer = BatchNormalization(momentum=CNN.MOMENTUM, name=(mdlname + "_BN1"))(layer)
    layer = Activation(activation=CNN.ACTIVATION, name=(mdlname + "_Activation_Conv3D"))(layer)

    layer = Conv1D(filters=this_num_filters, kernel_initializer=CNN.WT_INITIALIZER,
                   kernel_size=(CNN.FILTER_1_SIZE,), strides=CNN.STRIDE_1,
                   name=(mdlname + "_Conv1D_stride"))(layer)
    layer = BatchNormalization(momentum=CNN.MOMENTUM, name=(mdlname + "_BN_stride"))(layer)
    layer = Activation(activation=CNN.ACTIVATION, name=(mdlname + "_Activation_stride"))(layer)

    # each block consists of CONV1D - BatchNormalization - RELU
    for jblock in range(this_depth_1D // CNN.BLOCK_REPETITION):
        identity = layer  # create a copy of layer
        for jconv1D in range(CNN.BLOCK_REPETITION):
            layer = _CONV1D_BN_RELU(this_input=layer, filters=this_num_filters,
                                    temporal_filter_size=this_temporal_filter_size,
                                    mdlname=(mdlname + "_" + str(jblock) + str(jconv1D)))

        layer = _shortcut(this_input=identity, residual=layer, mdlname=(mdlname + "_" + str(jblock) + str(jconv1D)))

    # 1x1 convolution blocks
    for kblock in range(CNN.CONV1X1_BLOCK):
        identity = layer
        for kconv1D in range(2):
            layer = Conv1D(filters=this_num_filters, kernel_initializer=CNN.WT_INITIALIZER,
                           kernel_size=(1,),
                           name=(mdlname + "Conv1D_1x1" + "_" + str(kblock) + str(kconv1D)))(layer)
            layer = BatchNormalization(momentum=CNN.MOMENTUM,
                                       name=(mdlname + "_BN_1x1" + "_" + str(kblock) + str(kconv1D)))(layer)
            layer = Activation(activation=CNN.ACTIVATION,
                               name=(mdlname + "_Activation_1x1" + "_" + str(kblock) + str(kconv1D)))(layer)

        # outdim doesn't change when using 1x1 convolution
        layer = add([identity, layer])

    layer_avg = Lambda(lambda arg: K.mean(arg, axis=1), name=(mdlname + "_avg"))(layer)
    layer = Dense(outputdim)(layer_avg)

    return layer


def tf_model_imfcsnet(outputdim=1):
    this_input = Input(shape=(None, None, None, 1))

    layer = gen_model(outputdim, this_input, 0)
    model = Model(inputs=this_input, outputs=layer)
    return model


if __name__ == "__main__":
    testmdl = tf_model_imfcsnet(1)
    testmdl.summary()

    testdata = np.random.random(size=(2, 2500, 3, 3))
    testdata = np.expand_dims(testdata, axis=-1)

    # returning only mu's
    out1 = testmdl.predict(x=testdata, steps=1)
    print(out1)
