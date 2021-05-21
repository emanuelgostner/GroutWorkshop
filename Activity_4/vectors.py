# ------------------------------------------------------------------
# File name       : vectors.py
# ------------------------------------------------------------------
# File description:
# Vector arithmetic using TensorFlow
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf
import numpy as np
import sys
import time


# ------------------------------------------------------
# def print_np(value)
# ------------------------------------------------------

def print_np(value):

    print(' Print NumPy array ----------------------------------------')
    print(value)
    print(type(value))
    print(value.shape)
    print(value.size)
    print(value.dtype)


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

    print('\n-----------------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('---------------------------------------------------------\n')

    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    a = tf.constant(2.0, name='a', dtype=tf.float32)
    b = tf.Variable([1, 2, 3])
    c = tf.Variable([4, 5, 6])
    d = np.array([9, 4, 5])
    e = tf.convert_to_tensor(d)
    e1 = tf.constant(d)
    e2 = tf.Variable(d)

    print('-- Part 1 -------------------------------------------------------')
    print_tf(a)
    print_tf(b)
    print_tf(c)
    print_np(d)
    print_tf(e)
    print_tf(e1)
    print_tf(e2)

    f = tf.math.add(b, e)
    g = tf.math.subtract(b, e)
    h = tf.math.multiply(b, e)

    j = tf.Variable(0.0, name='j', dtype=tf.float32)

    print('-- Part 2 -------------------------------------------------------')
    print(b.numpy())
    print(e.numpy())
    print(f.numpy())
    print(g.numpy())
    print(h.numpy())
    print(j.numpy())

    for i in range(0, tf.size(b)):
        j = tf.math.add(
            tf.cast(j, dtype=tf.float32),
            tf.math.multiply(tf.cast(b[i], dtype=tf.float32), tf.cast(c[i], dtype=tf.float32)))

    print('-- Part 3 -------------------------------------------------------')
    print(tf.size(j).numpy())
    print(tf.shape(j).numpy())
    print(j.dtype)
    print(j.numpy())

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('\n-----------------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('-----------------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
