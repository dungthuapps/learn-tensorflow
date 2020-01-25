"""Concept 1: MNIST and Simple Linear (DNN)

### Simple Linear

    1 Layer -> Loss -> Optimizer

### MNIST

- pixel arrays in gray-scale (0~1).
- 28 * 28
    -> can flatten to 1-D ~ (784,1) or (1, 784)
    -> or 2-D (28, 28, 1)
- 55000 images as a tensor (n-d array)

### Labels Encoding

- One-Hot Encoding
    - vector = n-classes, e.g 10
- means, 5500 labels ---converted ---> (10, 55000) 2-d array

### Procedures

    1. Load data with one-hot encoding
    2. First Insight of data
    3. Build Models
        1. PlaceHolders of Input X and output y_true
        2. Variable of Weights and Input
        3. Cost Function and Lost Function
        4. Optimization (Gradient Descent)
    4. Run model
        1. Specify epochs and batch size
        2. Init global variables (graphs)
        3. Create a session
        4. Inside session, create a feed_dict to place holders
            1. train = run session with feed_dict
        5. In another session, validation by
            1. pass another feed_dict = (test_input, test_output)
            2. validation = correction predictions of (y, and y_true)
            4. fit ~ run session

"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 1 - Loading data & Insights
#   Download mnist with one-hot encoding (zip files)
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
mnist.train.num_examples
mnist.test.num_examples

#   First insights
single_image = mnist.train.images[1].reshape(28, 28)
print(single_image.min(), single_image.max())
plt.imshow(single_image, cmap="gist_gray")
plt.show()

# 2 - Build Models
#   Placeholders of input
x = tf.placeholder(tf.float32, shape=[None, 784])

#   Variables of Weights and Bias
W = tf.Variable(tf.zeros([784, 10]))  # zeros init
b = tf.Variable(tf.zeros([10]))

#   Create Graph Operations
y = tf.matmul(x, W) + b

# ! 2b Cost and Loss Function
#   Placeholder for true output
y_true = tf.placeholder(tf.float32, [None, 10])  # one-hot-vector

#   Cost function of classification for n-classes
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y)
#   Lost func
L = tf.reduce_mean(cross_entropy)

# 2c Optimizer with Gradient Descent with step_size 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(L)

# 2d Create Session to run
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # ! Train
    epochs = 1000
    batch_size = 100
    for step in range(epochs):
        # Get next batch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_feed_dict = {x: batch_x, y_true: batch_y}
        sess.run(train, feed_dict=train_feed_dict)

    # ! Evaluation

    # y -> 10-bit vectors of prob
    #   --> argmax -> return index which has highest prob
    # y_true ~ [0, 1, 0, ..0]
    #   --argmax > also return index which == 1 (max)
    # if same index then true, else false
    _idx_y = tf.argmax(y, 1)    # n-bit vectors
    _idx_y_true = tf.argmax(y_true, 1)
    correct_predictions = tf.equal(_idx_y, _idx_y_true)

    # To calculate accuracy, convert bool -> 1s 0s
    correct_predictions = tf.cast(correct_predictions, tf.float32)

    # mean of 1s and 0s ~ sum (.) / n
    #   ~ % of correct_predictions
    accuracy = tf.reduce_mean(correct_predictions)

    # feed x -> get y
    # feed y_true -> get correct_predictions -> accuracy
    eval_feed_dict = {x: mnist.test.images, y_true: mnist.test.labels}
    acc_result = sess.run(accuracy, feed_dict=eval_feed_dict)
    print(acc_result)
