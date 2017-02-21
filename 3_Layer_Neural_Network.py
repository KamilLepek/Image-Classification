# 3-layer neural network (2 hidden layers) for the mnist data set

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# Parameters
step = 0.001  # Learning rate
batch_size = 128
number_of_iterations = 150

number_of_W1 = 512  # number of weights in layer 1
number_of_W2 = 512  # number of weights in layer 2
number_of_classes = 10  # there are ten classes of digits
size_of_input = 28*28  # 28x28 pixels is the size of every mnist image

# input data
X = tf.placeholder("float", [None, size_of_input])
Y = tf.placeholder("float", [None, number_of_classes])

# Creation of neural network
# tf.random_normal(shape)
# Weights and biases (W/B)
W1 = tf.Variable(tf.random_normal([size_of_input, number_of_W1]))
B1 = tf.Variable(tf.random_normal([number_of_W1]))
# layer 1 = input*W1+B1
W2 = tf.Variable(tf.random_normal([number_of_W1, number_of_W2]))
B2 = tf.Variable(tf.random_normal([number_of_W2]))
# layer 2 = layer1*W2+B2
W3 = tf.Variable(tf.random_normal([number_of_W2,number_of_classes]))
B3 = tf.Variable(tf.random_normal([number_of_classes]))
# output = layer2*W3+B3

# Hidden layers will be activated with RELU
layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), B2))
output = tf.matmul(layer2, W3) + B3

# Cost/Loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))

# Optimizer
optimizer = tf.train.AdamOptimizer(step).minimize(cost)

# Launching
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initializing all variables
    for i in range(number_of_iterations):
        batch_numbers = int(mnist.train.num_examples / batch_size)
        cost_average = 0.
        for j in range(batch_numbers):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            opt, cost_temp = sess.run([optimizer, cost], feed_dict={X: batch_X, Y: batch_Y})
            cost_average += cost_temp/batch_numbers
        print("Iteration number %04d" % (i+1), "cost = {:.9f}".format(cost_average))
    print("Done")
    prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print("Accuracy: ", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
