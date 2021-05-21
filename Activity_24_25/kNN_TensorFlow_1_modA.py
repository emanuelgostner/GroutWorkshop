# ------------------------------------------------------------------
# Filename:    kNN_TensorFlow_1_modA.py
# ------------------------------------------------------------------
# File description:
# Python and TensorFlow image classification using the MNIST dataset.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------------------------------------
# Global variables
# ------------------------------------------------------


# ------------------------------------------------------
# def create_data_points()
# ------------------------------------------------------
# Create the data for two clusters (cluster 0 and cluster 1)
# Data points in cluster 0 belong to class 0 and data points in
# cluster 1 belong to class 1.
# ------------------------------------------------------
# x is the data point in the cluster and class_value is the class number
# Cluster : a cluster of data point values.
# Class   : the label of the class that the data point belongs to.
# ------------------------------------------------------

def create_data_points():

    # Cluster 0 data points (x0) / Class 0 label (class_value0 = 0)
    num_points_cluster0 = 100
    mu0 = [-0.5, 5]
    covar0 = [[1.5, 0], [0, 1]]

    # Generate random scattered points across given point by a given covariance.
    # mu0: the initial datapoint
    # covar0: Covariance indicates the level to which two variables vary together
    # num_points_cluster0: the amount of points that are generated
    x0 = np.random.multivariate_normal(mu0, covar0, num_points_cluster0)
    # Array with 0s that stand for the class 0 to each data point in x0
    class_value0 = np.zeros(num_points_cluster0)

    # Cluster 1 data points (x1) / Class 1 label (class_value1= 1)
    num_points_cluster1 = 100
    mu1 = [0.5, 0.75]
    covar1 = [[2.5, 1.5], [1.5, 2.5]]
    x1 = np.random.multivariate_normal(mu1, covar1, num_points_cluster1)
    class_value1 = np.ones(num_points_cluster1)

    return x0, class_value0, x1, class_value1


# ------------------------------------------------------
# def create_test_point_to_classify()
# ------------------------------------------------------

def create_test_point_to_classify():
    print('=====================================')
    print('User input for (x,y) point')
    print('=====================================')
    print('Enter number for x')
    while True:
        try:
            x = float(input())
            break
        except ValueError:
            print("The input is not valid, enter a number")

    print('Enter number for y')
    while True:
        try:
            y = float(input())
            break
        except ValueError:
            print("The input is not valid, enter a number")

    print('x value is now       -> %s' % str(x))
    print('y value is now       -> %s' % str(y))
    data_point = np.array([x, y])

    data_point_tf = tf.constant(data_point)

    return data_point, data_point_tf

# ------------------------------------------------------
# def user_input_k_value_tf()
# ------------------------------------------------------

def user_input_k_value_tf():
    print('=====================================')
    print('User input for k value')
    print('=====================================')
    print('Enter k value for algorithm. Number without decimals and between 0 and 200')
    while True:
        try:
            user_k_value = int(input())
            if user_k_value < 0 or user_k_value > 200:
                print("The input is not valid, enter a number between 0 and 200")
                continue
            break
        except ValueError:
            print("The input is not valid, enter a number")

    print('k value is now       -> %s' % str(user_k_value))
    return tf.constant(user_k_value)

# -------------------------------------------------------------------
# get_label(preds)
# -------------------------------------------------------------------

def get_label(preds):

    # print('-- Obtaining the class label')

    counts = tf.math.bincount(tf.dtypes.cast(preds, tf.int32))
    arg_max_count = tf.argmax(counts)

    # print('preds       -> %s' % str(preds))
    # print('counts      -> %s' % str(counts))
    # print('arg_max_count -> %s' % str(arg_max_count))

    return arg_max_count


# -------------------------------------------------------------------
# def predict_class(xt, ct, dt, kt)
# -------------------------------------------------------------------

def predict_class(xt, ct, dt, kt):

    neg_one = tf.constant(-1.0, dtype=tf.float64)
    distance = tf.reduce_sum(tf.abs(tf.subtract(xt, dt)), 1)

    neg_distance = tf.math.scalar_mul(neg_one, distance)
    val, val_index = tf.math.top_k(neg_distance, kt)
    cp = tf.gather(ct, val_index)

    return cp


# -------------------------------------------------------------------
# def plot_results(x0, x1, data_point, class_value)
# -------------------------------------------------------------------

def plot_results(x0, x1, data_point, class_value):

    plt.style.use('default')

    plt.plot(x0[:, 0], x0[:, 1], 'ro', label='class 0')
    plt.plot(x1[:, 0], x1[:, 1], 'bo', label='class 1')
    plt.plot(data_point[0], data_point[1], 'g', marker='D', markersize=10, label='Test data point')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Simple data point classification: Prediction is class %s' % class_value)
    plt.xlabel('Data x-value')
    plt.ylabel('Data y-value')

    plt.show()


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():

    print('=====================================')
    print('General Information')
    print('=====================================')
    print('This program creates 2 clusters/classes with 100 data points each.')
    print('The user provides and x,y and k value.')
    print('The program predicts the class of the user provided point(x,y) with the kNN Algorithm where the user input k is the amount of neighbors that are looked at for the prediciton')

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 1. Create the data points in each cluster (x0, class_value0, x1, class_value1)
    # 2. Create data point to classify (data_point, data_point_tf)
    # 3. Combine all cluster values into combined lists (x & class_value)
    # 4. Convert (x & class_value) values to TensorFlow constants (x_tf & class_value_tf)
    # ------------------------------------------------------
    while True:
        (x0, class_value0, x1, class_value1) = create_data_points()
        (data_point, data_point_tf) = create_test_point_to_classify()

        x = np.vstack((x0, x1))
        class_value = np.hstack((class_value0, class_value1))

        x_tf = tf.constant(x)
        class_value_tf = tf.constant(class_value)

        # ------------------------------------------------------
        # Run TensorFlow to predict the classification of data point and
        # print the predicted class using nearest 'k_value' data points.
        # ------------------------------------------------------
        k_value_tf = user_input_k_value_tf()
        pred = predict_class(x_tf, class_value_tf, data_point_tf, k_value_tf)
        class_value_index = pred
        class_value = get_label(class_value_index)
        print('=====================================')
        print('Prediction')
        print('=====================================')
        class_val = tf.keras.backend.get_value(class_value)

        print('Picked point %s is in class %s' % (str(data_point), class_val))
        print('=====================================')
        plot_results(x0, x1, data_point, class_value)

        print('Do you wish to repeat with new values? Press r to repeat, any other key to end the program')
        while True:
            try:
                end_program = input()
                if end_program == "r":
                    break
                else:
                    exit()
            except ValueError:
                print("Something went wrong")




# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
