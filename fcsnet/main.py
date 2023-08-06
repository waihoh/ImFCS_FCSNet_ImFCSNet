import os
import pickle
from Configurations import config as cfg
from Utilities.common_packages import np, tf
from nn_model.modelgenerator import tf_model, OnEpochEndCallback, modeltracker

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from Configurations import CNN

from simulator.fd1t_datagen import dg_train

# GPU setting
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0],'GPU')
tf.compat.v1.disable_v2_behavior()  # important. data generation process requires tf to be in non eager execution mode.


# training
def tf_start():
    # For saving training history to a .csv file.
    csv_logger = CSVLogger(cfg.FILENAME.tfModelPID + '_training.csv')


    # Learning rate decay.
    # See https://stackoverflow.com/questions/61552475/properly-set-up-exponential-decay-of-learning-rate-in-tensorflow
    learning_rate_decay_factor = (CNN.FINAL_LEARNING_RATE / CNN.INITIAL_LEARNING_RATE) ** (1 / CNN.TFEPOCHS)
    steps_per_epoch = CNN.BATCHES_PER_EPOCH

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=CNN.INITIAL_LEARNING_RATE,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    optimizer = Adam(learning_rate=lr_schedule)

    if CNN.IS_RETRAINING:
        model = load_model(CNN.retrain_model_fn, compile=False)
    else:
        model = tf_model(outputdim=1, model=CNN.NN_MODEL)  # currently predicting D only
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # write model.summary() to a text file instead of printing to console.
    with open('model_summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    cb_checkpoint = ModelCheckpoint(
        cfg.FILENAME.tfModelSaveFilepath,
        monitor='loss',
        save_weights_only=False,
        verbose=1,
        save_best_only=True, mode='min')

    # Set True to save only the last best model
    saved_fn_path = os.getcwd() + "/" + cfg.FILENAME.tfModelSaveFilepath
    only_last_model = modeltracker(
        fntemplate=saved_fn_path,
        onlylast=CNN.SAVE_ONLY_LAST_MODEL)

    cb_onepockendcallback = OnEpochEndCallback(
        datagenerator=dg_train,
        reachfullepoch=CNN.REACH_FULL_RANGE,
        tfepochs=CNN.TFEPOCHS,
        modeltracker=only_last_model)

    hist = model.fit(
        dg_train,
        epochs=CNN.TFEPOCHS,
        max_queue_size=dg_train.max_queue_size(),
        workers=dg_train.workers(),
        shuffle=dg_train.shuffle(),
        use_multiprocessing=dg_train.use_multiprocessing(),
        callbacks=[cb_checkpoint, cb_onepockendcallback, csv_logger])

    # Stop training and free up resources from GPU.
    dg_train.stop()

    # NOTE:
    # error "TypeError: Object of type 'float32' is not JSON serializable" when using json
    # using text file, some of the data is truncated when num epochs is large (eg. 200)
    # Thus, the saving of training history is now done with csv logger callback and pickle.
    fn = cfg.FILENAME.tfModelPID + "_hist.pickle"
    with open(fn, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

'''
MAIN
'''
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    tf_start()
