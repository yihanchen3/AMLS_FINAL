# Import packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv2D
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score
import numpy as np
from Modules.result_process import plot_history, plot_confusion_matrix


def CNNNET_2(input_shape):
    model = Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        # Fraction of the input units dropped
        Dropout(rate=0.5),
        # Number of units equal to the number of classes
        Dense(units=5, activation='softmax')
    ])
    model.summary()
    # Configures the model for training.
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def B1(input_shape, training_batches, valid_batches, test_batches, epochs=15, verbose=1, plot=True):
    model = CNNNET_2(input_shape)
    history = model.fit(x=training_batches,
                        steps_per_epoch=len(training_batches),
                        validation_data=valid_batches,
                        validation_steps=len(valid_batches),
                        epochs=epochs,
                        verbose=verbose
                        )
    if plot:
        # Plot loss and accuracy achieved on training and validation dataset
        plot_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                     history.history['val_loss'])

    # Steps parameter indicates how many batches are necessary to work on each data in the testing dataset
    # model.predict returns the predictions made on the input given
    # It returns the probabilities that each image belongs to the existing classes
    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=verbose)
    # Transform each prediction to an hot-encoding vector
    predictions = np.round(predictions)
    # The image is associated to the class with the highest probability
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    # Retrieve the true labels of the input
    true_labels = np.array(test_batches.classes)
    # Plot results through a confusion matrix
    plot_confusion_matrix('auto', predicted_labels, true_labels)

    train_accuracy = history.history['acc'][-1]
    valid_accuracy = history.history['val_acc'][-1]
    test_accuracy = accuracy_score(true_labels, predicted_labels)

    # Return accuracy on the test dataset   Return accuracy on the train and validation dataset
    return model, train_accuracy, valid_accuracy, test_accuracy
