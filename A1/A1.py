import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def allocate_weights_and_biases():
    # define number of hidden layers
    n_hidden_1 = 2048  # 1st layer number of neurons
    n_hidden_2 = 2048  # 2nd layer number of neurons

    # inputs placeholders
    X = tf.placeholder("float", [None, 68, 2])
    Y = tf.placeholder("float", [None, 2])  # 2 output classes

    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    # images_flat = tf.contrib.layers.flatten(X)
    images_flat = tf.compat.v1.layers.flatten(X)

    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01

    # define placeholders for weights and biases in the graph
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 2], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }

    return weights, biases, X, Y, images_flat


# Create model
def multilayer_perceptron():
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.sigmoid(layer_2)

    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer, X, Y


def A1(training_images, training_labels, test_images, test_labels, learning_rate=1e-5, training_epochs=300,
       display_accuracy_step=10):
    logits, X, Y = multilayer_perceptron()

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # define training graph operation
    train_op = optimizer.minimize(loss_op)

    # graph operation to initialize all variables
    init_op = tf.global_variables_initializer()
    accuracy_list = []
    cost_list = []
    with tf.Session() as sess:
        # run graph weights/biases initialization op
        sess.run(init_op)
        # begin training loop ..
        for epoch in range(training_epochs):
            # complete code below
            # run optimization operation (backprop) and cost operation (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))

            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
             # calculate training accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            train_accuracy = accuracy.eval({X: training_images, Y: training_labels})
            print("Accuracy: {:.4f}".format(train_accuracy))

            accuracy_list.append(train_accuracy)
            cost_list.append(cost)

        print("Optimization Finished!")
        # -- Define and run test operation -- #
        fig = plt.figure()
        fig.suptitle('Accuracy & Loss changes during training ')
        plt.subplot(211)
        plt.ylabel('Accuracy')
        plt.plot(accuracy_list)
        plt.subplot(212)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.plot(cost_list)
        plt.show()

        # apply softmax to output logits
        pred = tf.nn.softmax(logits)

        #  derive inferred classes as the class with the top value in the output density function
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # run test accuracy operation ..
        test_accuracy = accuracy.eval({X: test_images, Y: test_labels})
        print("Test Accuracy:", test_accuracy)
        return train_accuracy, test_accuracy
