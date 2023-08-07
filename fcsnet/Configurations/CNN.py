from Configurations import config as cfg

# ---------------------------------------------------------------
# DATA
EMCCD_FOLDER = "./training_data/data_EMCCD"
Gaussian_FOLDER = "./training_data/data_Gaussian"
PIXELS = 3  # simulated data are from ROI of 3X3 per image stack, i.e. 9 ACF curves per image stack
DATA_FN_START_INDEX = 1  # iterate through the data files. The file name format, e.g., 0001_x.npz, 0001_y.npz, ... 0013_x.npz, 0013_y.npz.
DATA_FN_COUNT = 13  # iterate through the data files, e.g., 0001_x.npz, 0001_y.npz, ... 0013_x.npz, 0013_y.npz, for each noise type.

# ---------------------------------------------------------------
# CONFIGURATIONS
GLOBALSEED = 1 # Change seed number for different initializations of FCSNet.

TFEPOCHS = 3000 
# there are 13 data files in each EMCCD and Gaussian data folders
# in the data generation process, there are 16 x 128, image stacks of size 3 x 3
# we shuffle and use a subset, i.e. 11, of the data: 1) to exclude a few ACFs where the 1st value is negative, and 2) to introduce some variation of data in each batch
BATCHES_PER_EPOCH = 16 * 11 * 2  # 2 for the two data folders 
BATCH_SIZE = 128 * PIXELS * PIXELS  # i.e. batch size of 1152

SAVE_ONLY_LAST_MODEL = True  # Intermediate models are deleted.
REACH_FULL_RANGE = 1

# ---------------------------------------------------------------
# RETRAINING
# NOTE: change the seed number too so that new data is generated when re-training a model
IS_RETRAINING = False  # set to True to reload a trained model for re-training
retrain_model_fn = ''  # If IS_RETRAINING, enter the .h5 file name of model to be retrained. The file should be placed in the root folder.

# ---------------------------------------------------------------
# DEFAULT SETTINGS
# see nn_model/fcsnet.py
NN_MODEL = cfg.EnumNNModels.fcsnet
ACTIVATION = 'relu'  # e.g. 'relu', 'sigmoid', 'tanh'
TEMPORAL_FILTER_SIZE = 18
SPATIAL_FILTER_SIZE = 3
DEPTH_1D = 4
BLOCK_REPETITION = 2
MOMENTUM = 0.9
NUM_FILTERS = 32
NUM_DENSE = 17

# For ReLU activation, use 'he_normal' weight initialization. See He et al 2015.
# For sigmoid or tanh activation, use default 'glorot_uniform'. See Glorot and Bengio 2010.
WT_INITIALIZER = 'he_normal'  # e.g. 'he_normal', 'glorot_uniform'

CONV1X1_BLOCK = 6

# Learning rates
INITIAL_LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE = 1e-5
