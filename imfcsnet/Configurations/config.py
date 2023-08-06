import os
from enum import Enum, auto


# scales of parameters
class EnumTrans(Enum):
    transNone = 0
    transLog = auto()
    transLogit = auto()
    transLogitAndSample = auto()


# indices of model parameters
class EnumModelParams(Enum):
    EmissionRate = 0
    ParticleDensity = auto()
    ParticleSig = auto()
    PhotonSig = auto()
    CCDNoiseRate = auto()


# indices of simulation parameters
class EnumSimParams(Enum):
    NumPixels = 0
    StepsPerFrame = auto()
    Margin = auto()
    Frames = auto()
    NoiseType = auto()
    EMCCDMin = auto()
    EMCCDFactorMin = auto()
    EMCCDFactorMax = auto()
    ZDimFactor = auto()
    ZFac = auto()
    LightSheetThickness = auto()


# type of model
class EnumNNModels(Enum):
    imfcsnet = 0


# Type of noise
class EnumNoiseType(Enum):
    Gaussian = 0  # Gaussian noise
    Experimental_EMCCD = 1  # experimental EMCCD camera probability mass function
    MixNoise = 2  # 50% Gaussian 50% experimental EMCCD


# file name
class FILENAME:
    # general format for saving model in .h5 format
    tfModelPID = "model-PID" + str(os.getpid())
    tfModelSaveFilepath = tfModelPID + "-fd1t-{epoch:04d}-{loss:.5f}.h5"

