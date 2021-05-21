import tensorflow as tf

def main():

    value1 = tf.constant(3.0, dtype=tf.float32)
    value2 = tf.constant(4.0)
    value3 = tf.math.add(value1,value2)

    print(value1)
    print(value2)
    print(value3)

if __name__ == '__main__':

    main()