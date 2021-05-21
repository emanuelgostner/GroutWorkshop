# ------------------------------------------------------------------
# File name       : TensorFlow_test_1.py
# ------------------------------------------------------------------
# File description:
# Test script to check TensorFlow installation.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
import sys
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():

    print('\n---------------------------------------------------')
    print('-- versions_cpu_gpu.py')
    print('---------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('---------------------------------------------------')
    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- NumPy version      : ' + str(np.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))
    print('---------------------------------------------------\n')

    # ------------------------------------------------------
    # Get list of local devices
    # ------------------------------------------------------

    print(device_lib.list_local_devices())

    local_devices = tf.config.experimental.list_physical_devices()

    # ------------------------------------------------------
    # CPUs
    # ------------------------------------------------------

    print('------------------------------------')
    print('Num CPUs Available: ', len(tf.config.experimental.list_physical_devices('CPU')))
    print('----------')

    for x in local_devices:
        if x.device_type == 'CPU':
            print(x)

    # ------------------------------------------------------
    # GPUs
    # ------------------------------------------------------

    print('------------------------------------')
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
    print('----------')

    for x in local_devices:
        if x.device_type == 'GPU':
            print(x)

    # ------------------------------------------------------
    # Script end actions
    # ------------------------------------------------------

    print('---------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('---------------------------------------------------')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
