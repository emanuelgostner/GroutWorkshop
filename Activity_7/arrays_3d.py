# ------------------------------------------------------------------
# File name       : arrays_3d.py
# ------------------------------------------------------------------
# File description:
# Arithmetic operations on 3D arrays using TensorFlow
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf
import sys
import time


# ------------------------------------------------------
# def print_tf(value)
# ------------------------------------------------------

def print_tf(value):

    print(' Print TensorFlow tensor ----------------------------------')
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
    print('-- TensorFlow version : ' + str(tf.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    a = tf.constant(2.0, name='a', dtype=tf.float32)

    b = tf.Variable([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ], dtype=tf.float32)

    c = tf.Variable([
        [[5, 2, 3], [10, 7, 2]],
        [[3, 5, 8], [12, 11, 6]]
    ], dtype=tf.float32)

    print('-- Part 1 -------------------------------------------------------')

    print_tf(a)
    print_tf(b)
    print_tf(c)

    print('-- Part 2 -------------------------------------------------------')

    d = tf.math.add(a, b)
    e = tf.math.add(b, c)
    f = tf.math.subtract(a, b)
    g = tf.math.subtract(b, c)
    h = tf.math.multiply(a, b)
    j = tf.math.multiply(b, c)

    print_tf(d)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_tf(e)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_tf(f)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_tf(g)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_tf(h)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_tf(j)

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
