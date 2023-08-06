from Configurations import config as cfg

###############################################################################
# SIMULATION MODEL PARAMETERS, i.e. the diffusion

# number of simulation model parameters
idxModTypes = len(cfg.EnumModelParams)

# NOTE: We divide EmissionRate by StepsPerFrame, i.e. 10, and ParticleSig by sqrt(StepsPerFrame), i.e. sqrt(10), in the data generation code. See fd1t_2d_datagen.py.
# base values of model parameters
mod = [0] * idxModTypes
mod[cfg.EnumModelParams.EmissionRate.value]        =  5.83       # average count rate per particle per frame
mod[cfg.EnumModelParams.ParticleDensity.value]     =  0.48374    # avg number of particles per pixel
mod[cfg.EnumModelParams.ParticleSig.value]         =  0.19185    # std of particle [pixel per frame]
mod[cfg.EnumModelParams.PhotonSig.value]           =  0.67012    
mod[cfg.EnumModelParams.CCDNoiseRate.value]        =  0.1        # variance of CCD Noise rate per frame per pixel, for Gaussian distribution


modTrans = [0] * idxModTypes
modTrans[cfg.EnumModelParams.EmissionRate.value]     =  cfg.EnumTrans.transNone
modTrans[cfg.EnumModelParams.ParticleDensity.value]  =  cfg.EnumTrans.transLog
modTrans[cfg.EnumModelParams.ParticleSig.value]      =  cfg.EnumTrans.transLog
modTrans[cfg.EnumModelParams.PhotonSig.value]        =  cfg.EnumTrans.transNone
modTrans[cfg.EnumModelParams.CCDNoiseRate.value]     =  cfg.EnumTrans.transNone


# parameters are chosen according to Gaussian, with mean = base value and sigma
# if the value is set to 0, then there is no randomness regardless modTrans.
# See randomParModel function in simulator > fd1t_2d_datagen.py
# If log-scale = True, sigma below is used as factor, so 1 means no randomness.
# or if modSig is set to 0, then no randomness.
modSig = [0] * idxModTypes
modSig[cfg.EnumModelParams.EmissionRate.value]     =  4.77         # e.g. CPS 1k to 10k, frame time 1.06 ms
modSig[cfg.EnumModelParams.ParticleDensity.value]  =  4.47214
modSig[cfg.EnumModelParams.ParticleSig.value]      =  7.07107      # e.g. range of D between 0.02 to 50 um^2/s
modSig[cfg.EnumModelParams.PhotonSig.value]        =  0.04188      # e.g. PSF 0.75 to 0.85, with mean of 0.8, with WL 583 nm and NA 1.45, , magnification 100X. It's uniformly sampled.
modSig[cfg.EnumModelParams.CCDNoiseRate.value]     =  0.01

# maximal values parameters are allowed to take, on natural scale
modMax = [0] * idxModTypes
modMax[cfg.EnumModelParams.EmissionRate.value]     =  100
modMax[cfg.EnumModelParams.ParticleDensity.value]  =  100
modMax[cfg.EnumModelParams.ParticleSig.value]      =  100
modMax[cfg.EnumModelParams.PhotonSig.value]        =  100
modMax[cfg.EnumModelParams.CCDNoiseRate.value]     =  100

# minimal values parameters are allowed to take, on natural scale
modMin = [0] * idxModTypes
modMin[cfg.EnumModelParams.EmissionRate.value]     =  0.
modMin[cfg.EnumModelParams.ParticleDensity.value]  =  0.
modMin[cfg.EnumModelParams.ParticleSig.value]      =  0.
modMin[cfg.EnumModelParams.PhotonSig.value]        =  0.
modMin[cfg.EnumModelParams.CCDNoiseRate.value]     =  0.

###############################################################################
# SIMULATION PARAMETERS

# number of simulation parameters
idxSimTypes = len(cfg.EnumSimParams)

# base values of simulation parameters
sim = [0] * idxSimTypes
sim[cfg.EnumSimParams.NumPixels.value]      =     3  # no. of pixels on the CCD
sim[cfg.EnumSimParams.StepsPerFrame.value]  =    10  # no. of sim. steps between frames
sim[cfg.EnumSimParams.Margin.value]         =     6  # no. of add. pixels simulated around CCD
sim[cfg.EnumSimParams.Frames.value]         =  2500  # no. of frames in one time series
sim[cfg.EnumSimParams.NoiseType.value]      =  cfg.EnumNoiseType.Gaussian.value  # type of camera noise during simulation, e.g.: .Gaussian, .Experimental_EMCCD, .MixNoise
sim[cfg.EnumSimParams.EMCCDMin.value]       =   -21  # minimum value of the distribution, after Hirsch correction of camera dark image.
sim[cfg.EnumSimParams.EMCCDFactorMin.value] =   0.1  # lower bound of the EMCCD noise (scaling) factor
sim[cfg.EnumSimParams.EMCCDFactorMax.value] =   0.5  # upper bound of the EMCCD noise (scaling) factor

# maximal values parameters are allowed to take, on natural scale
simMax = [0] * idxSimTypes
simMax[cfg.EnumSimParams.NumPixels.value] = sim[cfg.EnumSimParams.NumPixels.value]
simMax[cfg.EnumSimParams.StepsPerFrame.value] =  sim[cfg.EnumSimParams.StepsPerFrame.value]
simMax[cfg.EnumSimParams.Margin.value] = sim[cfg.EnumSimParams.Margin.value]
simMax[cfg.EnumSimParams.Frames.value] = sim[cfg.EnumSimParams.Frames.value]

# minimal values parameters are allowed to take, on natural scale
simMin = [0] * idxSimTypes
simMin[cfg.EnumSimParams.NumPixels.value] = sim[cfg.EnumSimParams.NumPixels.value]
simMin[cfg.EnumSimParams.StepsPerFrame.value] = sim[cfg.EnumSimParams.StepsPerFrame.value]
simMin[cfg.EnumSimParams.Margin.value] = sim[cfg.EnumSimParams.Margin.value]
simMin[cfg.EnumSimParams.Frames.value] = sim[cfg.EnumSimParams.Frames.value]


''' TEST: '''
if __name__ == "__main__":
    print('model parameters ' + str(len(cfg.EnumModelParams)))

    for i in range(len(cfg.EnumModelParams)):
        print(mod[i])

    print('simulation parameters')
    for i in range(len(cfg.EnumSimParams)):
        print(sim[i])

    for i in range(len(cfg.EnumModelParams)):
        print(modTrans[i])
