import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

X = tf.Variable([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
], dtype=tf.float32)

print('-- Part 1 ----------------------------------')
print(X)

print('-- Part 2 ----------------------------------')
print(tf.size(X))
print(tf.shape(X))
print(tf.size(tf.shape(X)))

print('-- X[0] -------------------')
print(X[0].numpy())
print('--------------')
print(X[0, 0].numpy())
print('--------------')
print(X[0, 1].numpy())
print('--------------')
print(X[0, 0, 0].numpy())
print(X[0, 0, 1].numpy())
print(X[0, 0, 2].numpy())
print(X[0, 1, 0].numpy())
print(X[0, 1, 1].numpy())
print(X[0, 1, 2].numpy())
print('---------------------------')
print('-- X[1] -------------------')
print(X[1].numpy())
print('--------------')
print(X[1, 0].numpy())
print('--------------')
print(X[1, 1].numpy())
print('--------------')
print(X[1, 0, 0].numpy())
print(X[1, 0, 1].numpy())
print(X[1, 0, 2].numpy())
print(X[1, 1, 0].numpy())
print(X[1, 1, 1].numpy())
print(X[1, 1, 2].numpy())

print('-- Part 3 ----------------------------------')
print(tf.size(X))
print(tf.shape(X))
print(tf.size(tf.shape(X)))

print('---------------------------')
for i in range(0, tf.shape(X)[0]):
    print(str(i) + ':\t' + str(X[i].numpy()))
print('---------------------------')
for i in range(0, tf.shape(X)[0]):
    for j in range(0, tf.shape(X)[1]):
        print(str(i) + ':' + str(j) + ':\t' + str(X[i, j].numpy()))
print('---------------------------')
for i in range(0, tf.shape(X)[0]):
    for j in range(0, tf.shape(X)[1]):
        for k in range(0, tf.shape(X)[2]):
            print(str(i) + ':' + str(j) + ':' + str(k) + ':\t' + str(X[i, j, k].numpy()))

print('--------------------------------------------')