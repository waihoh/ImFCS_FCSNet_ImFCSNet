from Configurations import config as cfg
from nn_model.fcsnet import tf_model_fcsnet
from tensorflow.keras.callbacks import Callback

import os

# for saving only the one/last best model.
class modeltracker():
    def __init__(self, fntemplate, onlylast=False):
        self.epoch = 0
        self.loss = float("inf")
        self.fntemplate = fntemplate
        self.onlylast = onlylast


def tf_model(outputdim=1, model=cfg.EnumNNModels):
    if model == cfg.EnumNNModels.fcsnet:
        return tf_model_fcsnet(outputdim=outputdim)
    else:
        raise SystemExit('Invalid neural network model')


# Custom callback
class OnEpochEndCallback(Callback):

    def __init__(self, datagenerator, reachfullepoch, tfepochs, modeltracker):
        super().__init__()
        self.datagenerator = datagenerator
        self.reachfullepoch = reachfullepoch
        self.tfepochs = tfepochs
        self.modeltracker = modeltracker

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if self.modeltracker.onlylast:
            # delete of previous saved model.
            current_loss = logs['loss']
            if current_loss < self.modeltracker.loss:
                fn = self.modeltracker.fntemplate.format(epoch=self.modeltracker.epoch,
                                                         loss=self.modeltracker.loss)
                if os.path.exists(fn):
                    os.remove(fn)
                self.modeltracker.loss = current_loss
                self.modeltracker.epoch = epoch + 1  # epoch here starts from index 0 while filename starts from index 1.
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
