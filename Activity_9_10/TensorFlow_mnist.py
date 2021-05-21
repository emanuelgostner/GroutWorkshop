# ------------------------------------------------------------------
# Filename:    TensorFlow_mnist.py
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


# ------------------------------------------------------
# def load_dataset(dataset)
# ------------------------------------------------------

def load_dataset(dataset):

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('>> Train: x=%s, y=%s' % (x_train.shape, y_train.shape))
    print('>> Test : x=%s, y=%s' % (x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


# ------------------------------------------------------
# def create_model()
# ------------------------------------------------------

def create_model():

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(512, activation='relu'),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.Dense(10, activation='softmax')]
        )

    return model  


# ------------------------------------------------------
# def compile_model(model)
# ------------------------------------------------------

def compile_model(model):
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )


# ------------------------------------------------------
# def fit_model(model, x_train, x_train,x_test, y_test, no_of_epochs)
# ------------------------------------------------------

def fit_model(model, x_train, y_train, x_test, y_test, no_of_epochs):

    history_callback = model.fit(
        x=x_train,
        y=y_train,
        epochs=no_of_epochs,
        validation_data=(x_test, y_test),
        )

    return history_callback


# ------------------------------------------------------
# def image_prediction(model, image)
# ------------------------------------------------------

def image_prediction(model, image):

    print('-- Image to predict')
    print('Shape of image ', image.shape)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(image.reshape(1, 28, 28))

    print(predictions)

    print('-- Predicted image')
    print('Most probable image ', tf.argmax(predictions[0]))


# ------------------------------------------------------
# def model_save(model, model_dir)
# ------------------------------------------------------

def model_save(model, model_dir):

    tf.keras.models.save_model(
        model,
        model_dir,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        )


# ------------------------------------------------------
# def model_load(model_dir)
# ------------------------------------------------------

def model_load(model_dir):

    reconstructed_model = tf.keras.models.load_model(
        model_dir,
        custom_objects=None,
        compile=True,
        )

    return reconstructed_model


# ------------------------------------------------------
# def plot_image(image_to_predict)
# ------------------------------------------------------

def plot_image(image_to_predict):

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    plt.suptitle('Image to predict')

    axs[0].imshow(image_to_predict)
    axs[0].set_title('Original image', fontsize=10)
    axs[0].set_xlabel('x pixel', fontsize=10)
    axs[0].set_ylabel('y pixel', fontsize=10)

    axs[1].imshow(image_to_predict, cmap=plt.get_cmap('gray'))
    axs[1].set_title('CMAP grayscale image', fontsize=10)
    axs[1].set_xlabel('x pixel', fontsize=10)
    axs[1].set_ylabel('y pixel', fontsize=10)
    
    axs[2].imshow(image_to_predict, cmap=plt.get_cmap('binary'))
    axs[2].set_title('CMAP binary image', fontsize=10)
    axs[2].set_xlabel('x pixel', fontsize=10)
    axs[2].set_ylabel('y pixel', fontsize=10)

    plt.tight_layout()
    plt.show()


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
    print('-- Matplotlib version : ' + str(mpl.__version__))

    dataset = tf.keras.datasets.mnist
    no_of_epochs = 5

    print('>> Number of epochs = ', no_of_epochs)

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    print('\n----------------------------------------------------------')
    print('-- 1. Load the dataset')
    print('----------------------------------------------------------\n')

    (x_train, y_train, x_test, y_test) = load_dataset(dataset)

    print('\n----------------------------------------------------------')
    print('-- 2. Create the model')
    print('----------------------------------------------------------\n')

    model = create_model()

    print('\n----------------------------------------------------------')
    print('-- 3. Compile the model')
    print('----------------------------------------------------------\n')

    compile_model(model)

    print('\n----------------------------------------------------------')
    print('-- 4. Train the model using the training data')
    print('----------------------------------------------------------\n')

    history_callback = fit_model(model, x_train, y_train, x_test, y_test, no_of_epochs)
    # history_callback.save("./")
    loss_history = history_callback.history['loss']
    accuracy_history = history_callback.history['accuracy']
    val_loss_history = history_callback.history['val_loss']
    val_accuracy_history = history_callback.history['val_accuracy']

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('loss         = ', loss_history)
    print('accuracy     = ', accuracy_history)
    print('val_loss     = ', val_loss_history)
    print('val_accuracy = ', val_accuracy_history)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    print('\n----------------------------------------------------------')
    print('-- 5. Evaluate the model using the test image set')
    print('----------------------------------------------------------\n')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('Test Loss:     ', test_loss)
    print('Test Accuracy: ', test_acc)

    print('\n----------------------------------------------------------')
    print('-- 6. Predict image from the test image set using the model')
    print('----------------------------------------------------------\n')

    test_image_id = tf.random.uniform((), minval=0, maxval=x_test.shape[0], dtype=tf.int32)
    image_to_predict = x_test[test_image_id]
    image_to_predict_label = y_test[test_image_id]

    print('** Image to predict1 is x_test[', test_image_id.numpy(), '] **')
    print('** This image has a label y_test[', image_to_predict_label, '] **')

    image_prediction(model, image_to_predict)

    print('\n----------------------------------------------------------')
    print('-- 7. Plot the image to predict')
    print('----------------------------------------------------------\n')

    print('Image to plot is x_test[', test_image_id.numpy(), ']')

    plot_image(image_to_predict)

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
