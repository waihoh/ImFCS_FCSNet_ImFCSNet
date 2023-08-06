from Configurations import config as cfg

# ---------------------------------------------------------------
# NOTE: 
# ImFCSNet training is essentially a single epoch training since all the data are generated and not re-used.
# There are 48000 (i.e. 1500 X 32) batches in one round of training.
# We partition the batches such that we decay the learning rate every 32 batches, and we'll have 1500 partitions
# BATCH_PARTITIONS is 1500 and BATCHES_PER_PARTITION is 32

# ---------------------------------------------------------------
# CONFIGURATIONS
IS_2D = True # set to True for 2D model or False for 3D model
GLOBALSEED = 1 # Change seed number for different initializations and generation of training data.

SAVE_ONLY_LAST_MODEL = True  # Intermediate models are deleted.
BATCH_PARTITIONS = 1500  # informatively, it corresponds to epoch/maxepochs in fd1t_2d_datagen.py, fd1t_3d_datagen.py, and main.py
BATCH_SIZE = 128 # It also corresponds to the number of threads (per block) in the numba code, see ufunc.py: ideally, it is 1024 to make use of all GPU threads per block.
BATCHES_PER_PARTITION = 32
REACH_FULL_RANGE = 1  # for gradual increase of range of parameters. 

# ---------------------------------------------------------------
# RETRAINING
# NOTE: change the seed number too so that new data is generated when re-training a model
IS_RETRAINING = False  # set to True to reload a trained model for re-training
retrain_model_fn = ''  # If IS_RETRAINING, enter the .h5 file name of model to be retrained. The file should be placed in the root folder.

# ---------------------------------------------------------------
# DEFAULT SETTINGS
# see nn_model/imfcsnet.py
NN_MODEL = cfg.EnumNNModels.imfcsnet
ACTIVATION = 'relu'  # e.g. 'relu', 'sigmoid', 'tanh'
TEMPORAL_FILTER_SIZE = 50
SPATIAL_FILTER_SIZE = 3
DEPTH_1D = 4
BLOCK_REPETITION = 2
MOMENTUM = 0.9
NUM_FILTERS = 45
NUM_DENSE = 17

# For ReLU activation, use 'he_normal' weight initialization. See He et al 2015.
# For sigmoid or tanh activation, use default 'glorot_uniform'. See Glorot and Bengio 2010.
WT_INITIALIZER = 'he_normal'  # e.g. 'he_normal', 'glorot_uniform'

#Conv3d layer
FILTER_3D_SIZE = 200

# 1st conv layer after conv3d.
FILTER_1_SIZE = 100
STRIDE_1 = 4

CONV1X1_BLOCK = 6

# Learning rates
INITIAL_LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE = 1e-5
