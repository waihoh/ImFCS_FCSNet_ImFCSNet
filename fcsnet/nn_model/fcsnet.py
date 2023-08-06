from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dense, Input, Cropping1D, add, Flatten
from tensorflow.keras.models import Model
from Configurations.fd1t_cfg import sim
from Configurations import config as cfg
from Configurations import CNN
from Utilities.common_packages import np, tf


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


def gen_model(outputdim, this_input, modelnum=0):
    # name of sub model
    mdlname = "mdl" + str(modelnum)

    # to add channel dimension and connect to the Conv1d layer
    layer = tf.expand_dims(this_input, axis=2, name=(mdlname + "_expand"))

    this_depth_1D = CNN.DEPTH_1D
    this_num_filters = CNN.NUM_FILTERS
    this_temporal_filter_size = CNN.TEMPORAL_FILTER_SIZE

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

    layer = Flatten()(layer)
    layer = Dense(outputdim)(layer)

    return layer


def tf_model_fcsnet(outputdim=1):
    this_input = Input(shape=sim[cfg.EnumSimParams.chanum.value])

    layer = gen_model(outputdim, this_input, 0)
    model = Model(inputs=this_input, outputs=layer)
    return model


if __name__ == "__main__":
    testmdl = tf_model_fcsnet(1)
    testmdl.summary()

    chanum = sim[cfg.EnumSimParams.chanum.value]
    print(f"num channels: {chanum}")  # 72 for correlatorP = 16 & correlatorQ =8
    testdata = np.random.random(size=(2, chanum))

    # returning only mu's
    out1 = testmdl.predict(x=testdata, steps=1)
    print(out1)
