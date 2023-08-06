import math
import threading
import time
import gc  # garbage collection
from Utilities.common_packages import np
from Configurations import config as cfg
from numba import float32, uint64, uint32, int64, cuda
from Utilities.ufunc import compile, randn, randu, randp
from Configurations.fd1t_cfg_3d import mod, modMax, modMin, modSig, modTrans, sim, simMax, simMin, idxModTypes, idxSimTypes
from Configurations import CNN
from .experimentalPMF import getEMCCDExperimentalCDF

from tensorflow.keras.utils import Sequence

# define indexes
idxSimNumPixels = cfg.EnumSimParams.NumPixels.value
idxSimStepsPerFrame = cfg.EnumSimParams.StepsPerFrame.value
idxSimMargin = cfg.EnumSimParams.Margin.value
idxSimFrames = cfg.EnumSimParams.Frames.value
idxSimNoiseType = cfg.EnumSimParams.NoiseType.value
idxSimEMCCDMin = cfg.EnumSimParams.EMCCDMin.value
idxSimEMCCDFactorMin = cfg.EnumSimParams.EMCCDFactorMin.value
idxSimEMCCDFactorMax = cfg.EnumSimParams.EMCCDFactorMax.value

idxSimZDimFactor = cfg.EnumSimParams.ZDimFactor.value
idxSimZFac = cfg.EnumSimParams.ZFac.value
idxSimLightSheetThickness = cfg.EnumSimParams.LightSheetThickness.value

idxModParticleDensity = cfg.EnumModelParams.ParticleDensity.value
idxModEmissionRate = cfg.EnumModelParams.EmissionRate.value
idxModCCDNoiseRate = cfg.EnumModelParams.CCDNoiseRate.value
idxModParticleSig = cfg.EnumModelParams.ParticleSig.value
idxModPhotonSig = cfg.EnumModelParams.PhotonSig.value

# set transformation values
transNone = cfg.EnumTrans.transNone
transLog = cfg.EnumTrans.transLog
transLogit = cfg.EnumTrans.transLogit
transLogitAndSample = cfg.EnumTrans.transLogitAndSample

# noise type
Gaussian_noise = cfg.EnumNoiseType.Gaussian.value
Experimental_EMCCD_noise = cfg.EnumNoiseType.Experimental_EMCCD.value
Mix_noise = cfg.EnumNoiseType.MixNoise.value


def simulateScanFunc(rng, fromSeqId, toSeqId, seqId, scan, modelPar, simPar, position, EMCCD, numFrames):
    # simulation parameters
    numPixels = simPar[seqId, idxSimNumPixels]
    numSteps = simPar[seqId, idxSimStepsPerFrame]
    numMargin = simPar[seqId, idxSimMargin]
    numFrames = numFrames[seqId]  # simPar[seqId, idxSimFrames]
    NoiseType = int64(simPar[seqId, idxSimNoiseType])

    simZDimFactor = float32(simPar[seqId, idxSimZDimFactor])
    simZFac = float32(simPar[seqId, idxSimZFac])
    simLightSheetThickness = float32(simPar[seqId, idxSimLightSheetThickness])

    ThicknessLL = -simZDimFactor * simLightSheetThickness
    ThicknessUL = simZDimFactor * simLightSheetThickness
    totalThickness = ThicknessUL - ThicknessLL

    # derived simulation parameters
    numPixelsf32 = float32(numPixels)
    leftBound = float32(-numMargin)
    rightBound = float32(numPixels + numMargin)
    totalWidth = rightBound - leftBound

    # model parameters (step adjusted)
    numParticles = uint64(modelPar[seqId, idxModParticleDensity] * totalWidth * totalWidth)
    emissionRatePerStep = float32(modelPar[seqId, idxModEmissionRate] / numSteps)
    CCDNoiseRate = float32(modelPar[seqId, idxModCCDNoiseRate])
    particleSigPerStep = float32(modelPar[seqId, idxModParticleSig] / math.sqrt(float32(numSteps)))
    photonSig = float32(modelPar[seqId, idxModPhotonSig])

    EMCCD_CDF = EMCCD[seqId, :]
    EMCCD_nMin = float32(simPar[seqId, idxSimEMCCDMin])
    EMCCD_FactorMin = float32(simPar[seqId, idxSimEMCCDFactorMin])
    EMCCD_FactorMax = float32(simPar[seqId, idxSimEMCCDFactorMax])

    EMCCD_scale_factor = float32(randu(rng, seqId) * (EMCCD_FactorMax - EMCCD_FactorMin) + EMCCD_FactorMin)

    if NoiseType == int64(Gaussian_noise):
        IsGaussianNoise = True
    elif NoiseType == int64(Experimental_EMCCD_noise):
        IsGaussianNoise = False
    else:
        IsGaussianNoise = float32(randu(rng, seqId)) >= 0.5
# NOTE: Cannot use else: raise SystemExit ... with numba. Error encountered.
#    else:
#        raise SystemExit("Invalid Noise Type in simulationScanFunc")

    # initialize particle positions
    for particleId in range(numParticles):
        position[seqId, particleId, 0] = randu(rng, seqId) * totalWidth + leftBound
        position[seqId, particleId, 1] = randu(rng, seqId) * totalWidth + leftBound
        position[seqId, particleId, 2] = randu(rng, seqId) * totalThickness + ThicknessLL

    # start main loop
    for frame in range(numFrames):
        for pixelX in range(numPixels):
            for pixelY in range(numPixels):
                if IsGaussianNoise:
                    scan[seqId, frame, pixelX, pixelY, 0] = math.sqrt(CCDNoiseRate) * randn(rng, seqId)
                else:
                    # using inverse transform sampling iso rejection sampling.
                    # See https://en.wikipedia.org/wiki/Inverse_transform_sampling.
                    unifval = float32(randu(rng, seqId))
                    counter = uint32(0)
                    while float32(EMCCD_CDF[counter]) < unifval:
                        counter += uint32(1)
                    scan[seqId, frame, pixelX, pixelY, 0] = EMCCD_scale_factor * (float32(counter) + EMCCD_nMin)

        for particleId in range(numParticles):
            posx = position[seqId, particleId, 0]
            posy = position[seqId, particleId, 1]
            posz = position[seqId, particleId, 2]

            for step in range(numSteps):
                posx += particleSigPerStep * randn(rng, seqId)
                posy += particleSigPerStep * randn(rng, seqId)
                posz += particleSigPerStep * randn(rng, seqId)

                # check if particle is outside the simulation area
                if posx < leftBound or posx > rightBound \
                        or posy < leftBound or posy > rightBound \
                        or posz > ThicknessUL or posz < ThicknessLL:

                    newpos = randu(rng, seqId) * totalWidth + leftBound
                    side = randu(rng, seqId)
                    side_on_z = randu(rng, seqId)  # choose if we place the particle on z boundaries, else it is on either x or y boundaries.

                    if side_on_z < 0.5:
                        # random on z-axis. place particle on one of the boundaries along either x-axis or y-axis.
                        if side < 0.25:
                            posx = newpos
                            posy = leftBound
                        elif side < 0.50:
                            posx = newpos
                            posy = rightBound
                        elif side < 0.75:
                            posx = leftBound
                            posy = newpos
                        else:
                            posx = rightBound
                            posy = newpos
                        posz = randu(rng, seqId) * totalThickness + ThicknessLL
                    else:
                        # place particle on either boundary on z-axis. random on x-axis and y-axis
                        if side < 0.5:
                            posz = ThicknessLL
                        else:
                            posz = ThicknessUL
                        posx = newpos
                        posy = randu(rng, seqId) * totalWidth + leftBound

                # NOTE:
                # There's a factor 1/2 in the calculation of zcor, which is different from the equation in the paper.
                # It is because the 1/e^2 radius of a 2D Gaussian to approximate the microscope PSF (in x-y plane) in the ImFCS ImageJ 3D simulation code was derived based on confocal definition/notation.
                # The code here is based on the ImFCS ImageJ 3D simulation code.
                zcor = photonSig + math.fabs(posz) * simZFac / 2

                # sample number of photons emitted during this time step
                photons = randp(rng, seqId, emissionRatePerStep)
                photons = uint32(math.floor(math.fabs(photons * math.exp(-0.5 * (posz / simLightSheetThickness) ** 2)) + 0.5))

                for _ in range(photons):
                    # calculate location of photon
                    locx = posx + zcor * randn(rng, seqId)
                    locy = posy + zcor * randn(rng, seqId)
                    # if photon left detector field
                    if locx < 0.0 or locx >= numPixelsf32 or locy < 0.0 or locy >= numPixelsf32:
                        continue
                    # photon within detector field, so calculate corresponding pixel
                    pixelX = uint32(locx)
                    pixelY = uint32(locy)
                    # update scan count
                    scan[seqId, frame, pixelX, pixelY, 0] += 1.0
            # store positions back to main array
            position[seqId, particleId, 0] = posx
            position[seqId, particleId, 1] = posy
            position[seqId, particleId, 2] = posz


class DataGeneratorFD1T(Sequence):
    def __init__(self, batches, batchSize, maxepochs,
                 verbose=False,
                 targetmodelpars=[idxModParticleSig],
                 reachfullatepoch=1,
                 simulateScanFunc=simulateScanFunc):

        self.batches = batches
        self.batchSize = batchSize
        self.maxepochs = maxepochs
        self.targetmodelpars = targetmodelpars
        self.reachfullatepoch = reachfullatepoch
        self.verbose = verbose
        self.data_regen = True

        self.epochs = 1
        self.numSeq = batches * batchSize
        self.maxPixels = simMax[idxSimNumPixels]
        self.maxFrames = int64(simMax[idxSimFrames])
        self.minFrames = int64(simMin[idxSimFrames])
        self.numFrames = self.minFrames
        self.maxWidth = float32(simMax[idxSimNumPixels] + 2 * simMax[idxSimMargin])
        self.maxParticles = uint64(modMax[idxModParticleDensity] * self.maxWidth * self.maxWidth)

        self.idxModTypes = idxModTypes
        self.mod = mod
        self.modSig = modSig
        self.modTrans = modTrans
        self.modMax = modMax
        self.modMin = modMin
        self.idxSimTypes = idxSimTypes
        self.sim = sim
        self.simMax = simMax
        self.simMin = simMin

        self.x = np.empty((self.numSeq, self.minFrames, self.maxPixels, self.maxPixels, 1), dtype='float32')
        self.y = np.empty((self.numSeq, len(self.targetmodelpars)), dtype='float32')
        self.modelPar = np.empty((self.numSeq, self.idxModTypes), dtype='float32')
        self.simPar = np.empty((self.numSeq, self.idxSimTypes), dtype='float32')
        self.position = np.empty((self.numSeq, self.maxParticles, 3), dtype='float32')  # 3D

        # Get experimental EMCCD noise probability mass function.
        temp_emccd_cdf = getEMCCDExperimentalCDF()
        self.EMCCD_CDF = np.empty((self.numSeq, temp_emccd_cdf.size), dtype='float32')
        for i in range(self.numSeq):
            self.EMCCD_CDF[i, :] = float32(temp_emccd_cdf)

        self.simulateScanFunc = compile(self.numSeq)(simulateScanFunc)

        self.ready = np.zeros(self.batches, dtype='bool')
        self.pleasestop = False
        self.gpumonitorthread = threading.Thread(target=self.gpumonitor)
        self.gpumonitorthread.start()

    def info(self):
        print("\n\n\n\n*******************************************************")
        print('model parameters dimensions:', self.modelPar.shape)
        print('simulation parameters dimensions:', self.simPar.shape)
        print('min x dimensions:', self.x.shape)
        print('y dimensions:', self.y.shape)
        print('position dimensions:', self.position.shape)
        print("*******************************************************\n\n\n\n")

    def __len__(self):
        return self.batches

    def stop(self):
        self.pleasestop = True

    def gpumonitor(self):
        while not self.pleasestop:
            while not self.data_regen:
                # to prevent CUDA_OUT_OF_MEMORY error. previous self.x and self.y might still be in used for training.
                time.sleep(5)

            while self.ready.any():
                if self.pleasestop:
                    print("---------------------- STOPPING ----------------------")
                    return
                time.sleep(5)

            # data will be generated here. set the value to false.
            # It is set to True in nn_model > modelgenerator.py > on_epoch_end function
            self.data_regen = False

            # NOTE: See https://github.com/numba/numba/issues/1531
            # to prevent CUDA_OUT_OF_MEMORY error.
            # Although it is observed that existing code does continue running after error message is shown.
            cuda.current_context().deallocations.clear()

            if self.epochs <= self.maxepochs:
                self.randomParModel(0, self.numSeq)
                self.randomParSim(0, self.numSeq)
                if self.verbose:
                    print("\n\n\nGPU Mode: In gpumonitor. Generating new set of data")

                self.numFrames = self.minFrames + int64((self.maxFrames - self.minFrames) * np.random.uniform())
                numFramesArr = np.full(shape=(self.numSeq,), fill_value=self.numFrames, dtype='float32')

                self.x = np.empty((self.numSeq, self.numFrames, self.maxPixels, self.maxPixels, 1), dtype='float32')
                self.simulateScanFunc(0, self.numSeq, self.x, self.modelPar, self.simPar, self.position,
                                        self.EMCCD_CDF, numFramesArr)
                if self.verbose:
                    print("\n\n\nGPU Mode: In gpumonitor. Done with new set of data")

                # normalize scan
                for seqId in range(self.numSeq):
                    avg = self.x[seqId, ].mean()
                    std = self.x[seqId, ].std()
                    self.x[seqId, ] -= avg
                    self.x[seqId, ] /= std
                    for tpi in range(len(self.targetmodelpars)):
                        z = self.modelPar[seqId, self.targetmodelpars[tpi]]
                        if modTrans[self.targetmodelpars[tpi]] == transLog:
                            self.y[seqId, tpi] = math.log(z)
                        elif modTrans[self.targetmodelpars[tpi]] == transLogit:
                            self.y[seqId, tpi] = math.log(z / (1.0 - z))
                        else:
                            self.y[seqId, tpi] = z
                self.ready[:] = True
            else:
                self.pleasestop = True

    def __getitem__(self, idx):
        seqFrom, seqTo = idx * self.batchSize, (idx + 1) * self.batchSize

        while not self.ready[idx]:
            time.sleep(5)
        self.ready[idx] = False

        dataout = self.x[seqFrom:seqTo, 0:self.numFrames, ...]
        target = self.y[seqFrom:seqTo, 0:self.numFrames, ...]

        if self.verbose and idx == 0:
            print("\n\n\nIn getitem. Some random model parameters at epoch", self.epochs)
            print(self.modelPar[0:10])
            print("\nSome x data:")
            print(self.x[0, 0, :, :, 0])
            print("\nSome y data:")
            print(self.y[0:10])
            print("\n")

        return dataout, target


    def randomParModel(self, seqFrom, seqTo):
        if seqFrom < 0 or seqTo > self.modelPar.shape[0]:
            raise ValueError("index out of bound")
        if self.modelPar.shape[1] < self.idxModTypes:
            raise ValueError("array 'modelPar' not big enough to hold all model parameters")
        for ind in range(self.idxModTypes):
            rnd = (2.0 * np.random.rand(seqTo - seqFrom) - 1.0).astype(self.modelPar.dtype)
            if not self.modSig[ind] == 0.0:
                if self.modTrans[ind] == transLog:
                    sig = math.log(self.modSig[ind])
                elif self.modTrans[ind] in [transLogit, transLogitAndSample]:
                    sig = math.log(self.modSig[ind] / (1 - self.modSig[ind]))
                else:
                    sig = self.modSig[ind]
                if not ind in self.targetmodelpars and self.epochs < self.reachfullatepoch:
                    sig *= (self.epochs / self.reachfullatepoch)**2
            else:
                sig = 0.0
            rnd *= sig

            if self.modTrans[ind] == transLog:
                rnd += math.log(self.mod[ind])
                np.exp(rnd, out=rnd)
            elif self.modTrans[ind] in [transLogit, transLogitAndSample]:
                rnd += math.log(self.mod[ind] / (1 - self.mod[ind]))
                np.exp(rnd, out=rnd)
                rnd /= 1.0 + rnd
            else:
                rnd += mod[ind]
            np.clip(rnd, a_min=self.modMin[ind], a_max=self.modMax[ind], out=rnd)
            if self.modTrans[ind] == transLogitAndSample:
                rnd[:] = np.random.rand(seqTo - seqFrom) < rnd
            self.modelPar[seqFrom:seqTo, ind] = rnd

    def randomParSim(self, seqFrom, seqTo):
        if seqFrom < 0 or seqTo > self.simPar.shape[0]:
            raise ValueError("index out of bound")
        if self.simPar.shape[1] < self.idxSimTypes:
            raise ValueError("array 'modelPar' not big enough to hold all model parameters")
        for i in range(self.idxSimTypes):
            self.simPar[seqFrom:seqTo, i] = self.sim[i]

    def on_epoch_end(self):
        if self.verbose:
            print("\n\n\non_epoch_end() called (", self.epochs, " done )")
        self.epochs += 1
        gc.collect()
        self.data_regen = True

    def max_queue_size(self):
        return 0

    def workers(self):
        return 1

    def use_multiprocessing(self):
        return False

    def shuffle(self):
        return False


targetmodelpars = [cfg.EnumModelParams.ParticleSig.value]

dg_train_3d = DataGeneratorFD1T(batches=CNN.BATCHES_PER_PARTITION, batchSize=CNN.BATCH_SIZE, reachfullatepoch=CNN.REACH_FULL_RANGE,
                                maxepochs=CNN.BATCH_PARTITIONS, verbose=True, targetmodelpars=targetmodelpars)

