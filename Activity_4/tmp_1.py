import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1, 2, 3, 4])
b = tf.constant([5, 2, 1, 5])
c = tf.math.add(a, b)

print(a)
print(b)
print(c)
print(tf.size(a))