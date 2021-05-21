# ------------------------------------------------------------------
# File name       : tensorFlow_test_1.py
# ------------------------------------------------------------------
# File description:
# Test script to check TensorFlow installation.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf
import sys
import time


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

    hello = tf.constant('A string constant in TensorFlow')

    print('-- Part 1 ----------------------------------\n')
    print('hello\t\t\t=>\t', hello, '\n')

    value1 = tf.constant(3.0, dtype=tf.float32)  # Number is a float32
    value2 = tf.constant(4.0)  # tf.float32 implicitly
    value3 = tf.math.add(value1, value2)
    value4 = tf.math.subtract(value1, value2)
    value5 = tf.math.multiply(value1, value2)
    value6 = tf.math.divide(value1, value2)
    value7 = tf.Variable(tf.math.divide(value1, value2))
    value8 = tf.Variable(1.0, dtype=float)

    a = tf.constant(3.0, dtype=tf.float32)  # Number is a float32
    b = tf.constant(4.0)
    x = tf.Variable([1, 2, 3, 4])
    x_float = tf.cast(x, dtype=tf.float32)
    c = tf.math.multiply(a, x_float)
    c_float = tf.cast(c, dtype=tf.float32)
    y = tf.math.add(c_float, b)

    print('-- Part 2 ----------------------------------\n')
    print('value1\t\t\t=>\t', value1)
    print('value2\t\t\t=>\t', value2)
    print('value3\t\t\t=>\t', value3)
    print('value4\t\t\t=>\t', value4)
    print('value5\t\t\t=>\t', value5)
    print('value6\t\t\t=>\t', value6)
    print('value7\t\t\t=>\t', value7)
    print('value8\t\t\t=>\t', value8)

    print('-- Part 3 ----------------------------------\n')
    print(a)
    print(b)
    print(c)
    print(c_float)
    print(x)
    print(x_float)
    print(y)
    print('-------------------------------------------\n')

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
