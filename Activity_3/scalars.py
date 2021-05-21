# ------------------------------------------------------------------
# File name       : scalars.py
# ------------------------------------------------------------------
# File description:
# Mathematical operations on scalars in TensorFlow
# ------------------------------------------------------------------
# References:
#     # https://www.tensorflow.org/api_docs/python/tf/math
#     # https://www.tensorflow.org/api_docs/python/tf/cast
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import sys
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


# ------------------------------------------------------
# def print_vals(value)
# ------------------------------------------------------

def print_vals(value):

    print(value)
    print(value.numpy())
    print(type(value))
    print(value.shape)
    print(tf.shape(value))
    print(tf.size(value))
    print(value.dtype)
    tf.print(value, output_stream=sys.stdout)


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():

    # ------------------------------------------------------
    # -- Start of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')

    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__) + '\n')

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    a = tf.constant(3, name='a')
    b = tf.constant(30.0, name='b')

    c = tf.Variable(5, name='c')
    d = tf.Variable(50., name='d')

    e = tf.math.add(tf.cast(a, dtype=float), b)
    f = tf.math.multiply(tf.cast(a, dtype=float), b)

    print('-- Part A ---------------------------------------')
    print_vals(a)
    print_vals(b)
    print_vals(c)
    print_vals(d)
    print('-- Part B ---------------------------------------')
    print_vals(e)
    print_vals(f)

    # ------------------------------------------------------
    # Exercise actions below - to be added in the script
    # file scalars_mod.py .
    # ------------------------------------------------------

    print('-- Part c ---------------------------------------')

    print('-------------------------------------------------\n')

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
