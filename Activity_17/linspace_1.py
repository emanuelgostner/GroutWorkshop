# Use TensorFlow to calculate y for the equation y = ax + b
# Use Matplotlib to plot the graph of y = ax + b
# https://www.tensorflow.org/api_docs/python/tf/linspace

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.constant(2.0)
b = tf.constant(5.0)

# 11 Werte zwischen 0 und 10
x = tf.linspace(0.0, 10.0, 11)
y = tf.Variable(tf.math.add(b, tf.math.multiply(a, x)))

print(a)
print(b)
print(x)
print(y)

plt.figure(facecolor='ivory', num='Example TensorFlow calculation plot')
ax = plt.axes()
ax.set_facecolor('palegreen')
plt.plot(x.numpy(), y.numpy(), '-k')
plt.plot(x.numpy(), y.numpy(), 'ro')
plt.grid()
plt.title('y = ax + b')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
