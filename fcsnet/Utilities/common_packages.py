from Configurations import CNN
import numpy as np
import tensorflow as tf
import random as random

'''
NOTE:
Get numpy, TensorFlow and random packages here and initialize random seeds. 
See also create_xoroshiro128p_states in ufunc.py.
'''

seed = CNN.GLOBALSEED

# Initialize seeds
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed=seed)
random.seed(a=seed)

# wt_glorot_uniform_initializer = tf.glorot_uniform_initializer(seed=seed)
