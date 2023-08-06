import os
from enum import Enum, auto


# indices of simulation parameters
class EnumSimParams(Enum):
    chanum = 0

# type of model
class EnumNNModels(Enum):
    fcsnet = 0

# file name
class FILENAME:
    # general format for saving model in .h5 format
    tfModelPID = "model-PID" + str(os.getpid())
    tfModelSaveFilepath = tfModelPID + "-fd1t-{epoch:04d}-{loss:.5f}.h5"
