#Consider the following equation: y = ax + b
#The following conditions exist:
# a can vary from 0 to +10 in steps of 1.
# b can vary from -5 to +5 in steps of 0.5.
# x varies from -10 to +10 in steps of 0.1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import random

a = tf.range(0., 10., 1, dtype=tf.float32)
b = tf.range(-5., 5., 0.5, dtype=tf.float32)
x = tf.range(-10., 10., 0.1, dtype=tf.float32)

with tf.GradientTape() as tape:
    a_rand = a.numpy()[random.randint(0, 9)]
    b_rand = b.numpy()[random.randint(0, 19)]
    tape.watch(x)
    # y = ax +b
    y = tf.add(
        tf.multiply(
            a_rand, x
        ),
        b_rand
    )

dy_dx = tape.gradient(y, x)

plt.grid()
plt.title('Plot of y = {}x + {} and dy/dx'.format(a_rand, b_rand))
plt.xlabel('x')
plt.ylabel('y (black) & dy/dx (red)')
plt.plot(x, y, 'k')
plt.plot(x, dy_dx, 'r')
plt.show()