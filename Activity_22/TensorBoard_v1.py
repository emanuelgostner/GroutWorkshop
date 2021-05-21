# ------------------------------------------------------------------
# File name       : TensorBoard_v1.py
# ------------------------------------------------------------------
# File description:
# TensorBoard example using TensorFlow in version 1 compatibility.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf

# ------------------------------------------------------
# Create the graph. Initially clear the default graph
# stack and reset the global default graph.
# ------------------------------------------------------

tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()

# ------------------------------------------------------
# Create tensors, constants, and variables.
# ------------------------------------------------------

a1 = tf.compat.v1.constant(2)
b1 = tf.compat.v1.constant(3)

a2 = tf.compat.v1.constant(5, name='a')
b2 = tf.compat.v1.constant(6, name='b')

c1 = tf.compat.v1.add(a1, b1)  # c1 = a1 + b1
c2 = tf.compat.v1.add(a2, b2, name='addition_c2')  # c2 = a2 + b2
c3 = tf.compat.v1.multiply(a2, c2, name='multiplication_c3')  # c3 = a2 * c2
c4 = tf.compat.v1.multiply(c1, c3, name='multiplication_c4')  # c4 = c1 * c3

x1 = tf.compat.v1.Variable(7, dtype=tf.int32)
x2 = tf.compat.v1.Variable(9, dtype=tf.int32)
x3 = tf.compat.v1.math.add(x1, x2)
x4 = tf.compat.v1.Variable(tf.compat.v1.math.add(a1, b1))

print(a1)
print(b1)
print(a2)
print(b2)
print(c1)
print(c2)
print(c3)
print(c4)
print(x1)
print(x2)
print(x3)
print(x4)

# ------------------------------------------------------
# Launch the graph in a session called 'sess'
# ------------------------------------------------------

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:

    sess.run(init)
    writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)

    print('c1 = ', sess.run(c1))
    print('c2 = ', sess.run(c2))

    d3 = sess.run(c3)
    print('d3 = ', d3)
    d4 = sess.run(c4)
    print('d4 = ', d4)
    d5 = sess.run(x1)
    print('d5 = ', d5)
    d6 = sess.run(x2)
    print('d6 = ', d6)
    d7 = sess.run(x3)
    print('d7 = ', d7)
    d8 = sess.run(x4)
    print('d8 = ', d8)

    writer.close()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------