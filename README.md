# Deep learning reduces data requirements and allows real-time measurements in Imaging Fluorescence Correlation Spectroscopy
This repository contains the training code of FCSNet and ImFCSNet models.


## Create the conda environment
Create a conda environment called *tf25_gpu* with the *environment.yml* file. The environment contains TensorFlow v2.5.

```
conda env create -f environment.yml
```

## Training FCSNet
To train FCSNet on a NVIDIA GPU, navigate to *fcsnet* folder and run the *main.py* file.
```
cd fcsnet
conda activate tf25_gpu
CUDA_VISIBLE_DEVICES=0 python main.py
```

The FCSNet training data can be placed in *training_data/data_EMCCD* and *training_data/data_Gaussian* folders. They are the simulated 2D/3D ACF curves with the respective noise type. You can change the seed and the hyperparameters of the network in *CNN.py* file to try different experiments.   

## Training ImFCSNet
Similarly, navigate to the *imfcsnet* folder and run the *main.py* file.

```
cd imfcsnet
conda activate tf25_gpu
CUDA_VISIBLE_DEVICES=0 python main.py
```

You can amend the configuration files, which are in the *Configurations* folder, to make changes to the ImFCSNet training. You can make the following changes to try different experiments.
1. In *CNN.py* file:
   * 2D/3D training: *IS_2D* variable
   * seed: *GLOBALSEED* variable
   * re-train a model: *IS_RETRAINING* and *retrain_model_fn* variables
   * learning rates: *INITIAL_LEARNING_RATE* and *FINAL_LEARNING_RATE* variables
   * configuration of the network

2. In *fd1t_cfg_2d.py* or *fd1t_cfg_3d.py* files, you can change the range of simulation 2D/3D parameters. 

## License

Distributed under the MIT license. See LICENSE for details.