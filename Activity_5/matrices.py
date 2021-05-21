# ------------------------------------------------------------------
# File name       : matrices.py
# ------------------------------------------------------------------
# File description:
# Matrix arithmetic using TensorFlow
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
    print(type(value)) # dtype des Tensors als Python reference
    print(value.dtype)  # dtype wie im Tensor definiert
    print(value.shape) # Shape in Form (x, y)
    print(tf.shape(value)) # Shape als Tensor
    print(tf.size(value)) # Anzahl aller Elemente im Tensor
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
    b = tf.Variable([[1, 2, 3], [4, 5, 6]])
    c = tf.Variable([[7, 8, 9], [10, 11, 12]])
    d = np.array([[6, 5 ,4], [9, 7, 1]])
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

    print('-- Part 2 -------------------------------------------------------')
    print(b.numpy())
    print(e.numpy())
    print(f.numpy())
    print(g.numpy())
    print(h.numpy())

    print('-- Part 3 -------------------------------------------------------')

    j = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.int32)
    k = tf.constant([5, 6, 7, 8], shape=(2, 2), dtype=tf.int32)

    m = tf.multiply(j, k)
    n = tf.Variable(tf.multiply(j, k))
    p = tf.matmul(j, k)
    q = tf.Variable(tf.matmul(j, k))

    print(j)
    print(k)
    print(m)
    print(n)
    print(p)
    print(q)

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
