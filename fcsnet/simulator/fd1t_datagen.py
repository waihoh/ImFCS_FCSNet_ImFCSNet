import threading
import time
import gc  # garbage collection
from Utilities.common_packages import np
from Configurations import config as cfg
# from Configurations.fd1t_cfg import sim
from Configurations import CNN

from tensorflow.keras.utils import Sequence

# define indexes
idxSimChaNum = cfg.EnumSimParams.chanum.value

class DataGeneratorFD1T(Sequence):
    def __init__(self, batches, batchSize, maxepochs,
                 verbose=False):

        self.batches = batches
        self.batchSize = batchSize
        self.maxepochs = maxepochs
        self.num_pred = 1  # currently we predict D only
        self.verbose = verbose
        self.get_new_data = True

        self.epochs = 1

        self.ready = np.zeros(self.batches, dtype='bool')
        self.pleasestop = False
        self.getdatathread = threading.Thread(target=self.getdata)
        self.getdatathread.start()

    def info(self):
        print("\n\n\n\n*******************************************************")
        print('x dimensions:', self.x.shape)
        print('y dimensions:', self.y.shape)
        print("*******************************************************\n\n\n\n")

    def __len__(self):
        return self.batches

    def stop(self):
        self.pleasestop = True

    def getdata(self):
        while not self.pleasestop:
            while not self.get_new_data:
                if self.pleasestop:
                    return                
                time.sleep(5)

            # load data once at the start
            if self.epochs == 1:
                start_digit = 1
                fns = [str(start_digit + i).zfill(4) for i in range(13)]
                all_data_x = []
                all_data_y = []                
                for fn in fns:
                    fn_x = fn + "_x.npz"
                    fn_y = fn + "_y.npz"                
                    # full EMCCD (without multiply any scaling factor)
                    filepath_x = CNN.EMCCD_FOLDER + "/" + fn_x
                    filepath_y = CNN.EMCCD_FOLDER + "/" + fn_y
                    all_data_x.append(np.load(filepath_x)['arr_0'])
                    all_data_y.append(np.load(filepath_y)['arr_0'])
                    # Gaussian, variance 1 to 9
                    filepath_x = CNN.Gaussian_FOLDER + "/" + fn_x
                    filepath_y = CNN.Gaussian_FOLDER + "/" + fn_y
                    all_data_x.append(np.load(filepath_x)['arr_0'])
                    all_data_y.append(np.load(filepath_y)['arr_0'])
                
                self.x_main = np.concatenate(all_data_x, axis=0)
                self.y_main = np.concatenate(all_data_y, axis=0)

                self.x = np.reshape(self.x_main, newshape=(-1, self.x_main.shape[-1]))
                self.y = np.reshape(self.y_main, newshape=(-1, self.num_pred))

                # this filter away a few ACF curves that are "very noisy" with a simple condition that the 1st value is < 0
                condition1 = self.x[:, 0] > 0
                self.x = self.x[condition1]
                self.y = self.y[condition1]

                # normalization. The data are raw acf curves.
                ave_val = np.mean(self.x, axis=-1, keepdims=True)
                std_val = np.std(self.x, axis=-1, keepdims=True)
                self.x -= ave_val
                self.x /= std_val

                self.info()

            # Shuffle
            perm = np.random.permutation(len(self.x))
            self.x = self.x[perm]
            self.y = self.y[perm]

            self.ready[:] = True
            self.get_new_data = False

    def __getitem__(self, idx):
        seqFrom, seqTo = idx * self.batchSize, (idx + 1) * self.batchSize
        while not self.ready[idx]:
            time.sleep(5)
        self.ready[idx] = False

        # we simulate 3x3 image, i.e. we can get 9 ACF curves from each image stack
        dataout = self.x[seqFrom:seqTo, ...]
        outputs = self.y[seqFrom:seqTo, 0]

        if self.verbose and idx == 0:
            print("\n\n\nIn getitem. Some random model parameters at epoch", self.epochs)
            print("\nSome x data:")
            print(self.x[0:5, 0:5])
            print("\nSome y data:")
            print(self.y[0:5])
            print("\n")

        return dataout, outputs

    def on_epoch_end(self):
        if self.verbose:
            print("\n\n\non_epoch_end() called (", self.epochs, " done )")
        self.epochs += 1
        gc.collect()
        if self.epochs > self.maxepochs:
            self.pleasestop = True
        else:        
            self.get_new_data = True

    def max_queue_size(self):
        return 0

    def workers(self):
        return 1

    def use_multiprocessing(self):
        return False

    def shuffle(self):
        return False


dg_train = DataGeneratorFD1T(batches=CNN.BATCHES_PER_EPOCH, batchSize=CNN.BATCH_SIZE, maxepochs=CNN.TFEPOCHS, verbose=True)

