# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets('../../data/mnist/', one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

   # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Train
  for idx in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss = sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
    print('step:%10d,loss:%10f,accuracy:%10f'%(idx, loss, acc))

import matplotlib.pyplot as plt
import string
def plot_log_data(filename):
    fid = open(filename,'r')
    accuracy = []
    loss = []
    step = []

    while True:
        line = fid.readline()
        if line == '':
            break;

        line_split = line.split(',')
        if len(line_split) != 3:
            print('plot_log_data: data error')
            continue

        pair_split = line_split[0].split(':')
        step_temp = string.atof(pair_split[1])
        pair_split = line_split[1].split(':')
        loss_temp = string.atof(pair_split[1])
        pair_split = line_split[2].split(':')
        accuracy_temp = string.atof(pair_split[1])


        accuracy.append(accuracy_temp)
        loss.append(loss_temp)
        step.append(step_temp)


    fid.close()
    #plot

    loss = [float(i)/max(loss) for i in loss]

    fig, ax = plt.subplots()

    ax2 = ax.twinx()
    # ax = fig.add_subplot(111)
    line1 = ax.plot(step, accuracy, color='red', label = 'accuracy')
    ax.legend(loc = 'upper right')


    line2 = ax2.plot(step, loss, color='blue', label = 'loss')
    ax2.set_ylabel('sin')
    ax2.legend(loc = 'upper left')
    plt.show()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    #                   help='Directory for storing input data')
    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    plot_log_data('mnist.txt')
