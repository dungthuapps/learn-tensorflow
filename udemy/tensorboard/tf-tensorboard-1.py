"""Abstraction: TensorBoard.

# Reference: 
    * https://www.tensorflow.org/tensorboard

# To activate tensorboard:
1. Go to CLI, activate virtualenv (if using it)
2. tensorboard --logdir="./output/"

"""

import tensorflow as tf


"""Tensorboard 1: Simple graph version 1."""

# Now go to CLI:
# activate the virtualenv (if you are using it)
#
# Notice: Linux path to dir different from Windows
# tensorboard --logdir="./output/"

tf.reset_default_graph()

# name will help to be clearer in graph vis
a = tf.add(1, 2)
b = tf.add(3, 4)
c = tf.multiply(a, b)


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output", sess.graph)
    print(sess.run(c))

    writer.close()

# Now go to CLI:
# activate the virtualenv (if you are using it)
#
# Notice: Linux path to dir different from Windows
# tensorboard --logdir="./output/"

"""Tensorboard 2: Simple graph version 2 (with Name of node)."""

tf.reset_default_graph()

# name will help to be clearer in graph vis
a = tf.add(1, 2, name="First_add")
b = tf.add(3, 4, name="Second_add")
c = tf.multiply(a, b, name="multiply")


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output", sess.graph)
    print(sess.run(c))

    writer.close()

"""Tensorborad 3: Simple graph version 3 (with Name, Scope)."""

# Scope
with tf.name_scope("Operation_A"):

    a = tf.add(1, 2, name="First_add")
    a1 = tf.add(100, 200, name="a_add")
    a2 = tf.multiply(a, a1)
    # Sub-Scope
    with tf.name_scope("Sub_Operation_SubA"):
        a3 = tf.add(a2, 100)

with tf.name_scope("Operation_B"):
    b = tf.add(3, 4, name="Second_add")
    b1 = tf.add(300, 400, name="b_add")
    b2 = tf.multiply(b, b1)

c = tf.multiply(a, b, name="multiply")


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output", sess.graph)
    print(sess.run(c))

    writer.close()

"""Tensorboard 4: Histogram, Summary"""
tf.reset_default_graph()
k = tf.placeholder(tf.float32)
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)

tf.summary.histogram(name="normal/moving_mean", values=mean_moving_normal)


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output", sess.graph)
    summaries = tf.summary.merge_all()

    N = 400
    for step in range(N):
        k_val = step / float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)

# When activate tensorboard in CLI
#   from localhost:6060 (default)
#       -> HISTOGRAM or DISTRIBUTION to see magic things
#
