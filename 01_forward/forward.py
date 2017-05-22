#!usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
#weight
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))
#input
# x = tf.constant([[0.7, 0.9],[0.8, 0.6]])
x = tf.placeholder(tf.float32, shape=(3,2), name="input")
#output of every layer
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(y,feed_dict={x:[[0.7, 0.9],[0.1, 0.4],[0.5, 0.8]]})

# sess = tf.Session()
# #init the all variable
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# #output
# print(sess.run(y))
#
# sess.close()

