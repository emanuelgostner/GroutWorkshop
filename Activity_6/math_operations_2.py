import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.range(-2., 6., 0.1, dtype=tf.float32)

with tf.GradientTape() as tape:

    tape.watch(x)

    y = tf.math.add(tf.math.add(tf.math.subtract(
        tf.pow(x, 3),
        tf.math.multiply(5., tf.math.square(x))),
        tf.math.multiply(2., x)),
        tf.constant(8.))

dy_dx = tape.gradient(y, x)

print('******************')
print('-- x --')
print(x)
print('******************')
print('-- y --')
print(y)
print('******************')
print('-- dy_dx --')
print(dy_dx)
print('******************')

plt.grid()
plt.title('Plot of y = x^3 - 5x^2 + 2x + 8 and dy/dx')
plt.xlabel('x')
plt.ylabel('y (black) & dy/dx (red)')
plt.plot(x, y, 'k')
plt.plot(x, dy_dx, 'r')
plt.show()