import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

k_value = tf.constant(3)

data_point = tf.constant([2., 3.])

dataset = tf.constant([[1, 1], [1, 4], [1, 5], [2, 2],
                       [2, 3], [2, 5], [4, 3], [5, 1]],
                      dtype=tf.float32)

neg_one = tf.constant(-1.0, dtype=tf.float32)
# liste mit den differenzen zwischen data_point und dataset values
distance = tf.reduce_sum(
    tf.abs(
        tf.subtract(dataset, data_point)), 1)
# mit -1 multiplizieren damit später bei top_k die niedristen values ganz oben rauskommen und zurückgegeben werden
neg_distance = tf.math.scalar_mul(neg_one, distance)
# val, val_index = tf.nn.top_k(neg_distance, k_value)
val, val_index = tf.math.top_k(neg_distance, k_value)

print('-- k_value -----------------------------')
print(k_value)
print('-- data_point --------------------------')
print(data_point)
print('-- dataset -----------------------------')
print(dataset)
print('-- neg_one -----------------------------')
print(neg_one)
print('-- distance ----------------------------')
print(distance)
print('-- neg_distance ------------------------')
print(neg_distance)
print('-- val ---------------------------------')
print(val)
print('-- val_index ---------------------------')
print(val_index)
print('----------------------------------------')
