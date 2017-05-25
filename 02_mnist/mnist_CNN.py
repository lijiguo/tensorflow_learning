# fileanme: mnist.py
# descripe: CNN for mnist
# author: lijiguo16@mails.ucas.ac.cn
# data: 20170523
# =====================

import tensorflow as tf
import numpy as np
import struct

DATAS_PATH = '../../data/mnist/'
DATAS_TRAIN_FILE_NAME = 'train-images-idx3-ubyte'
DATAS_TEST_FILE_NAME = 't10k-images-idx3-ubyte'
LABELS_TRAIN_FILE_NAME = 'train-labels-idx1-ubyte'
LABELS_TEST_FILE_NAME = 't10k-labels-idx1-ubyte'

DATAS_CLASS_NUM = 10

def _read32(content, offset):
    # dt = np.dtype(np.uint32).newbyteorder('>')
    # return np.frombuffer(content.read(4),dtype=dt)
    data = struct.unpack_from('>I',content,offset)[0]
    offset = offset + struct.calcsize('>I')
    return data, offset


def read_images_data(content):
    print('read images data...')
    offset = 0
    magic, offset = _read32(content, offset)
    if magic != 2051:
        raise ValueError('Invaild magic number in mnist image file')

    num_images, offset = _read32(content, offset)
    rows, offset = _read32(content, offset)
    cols, offset = _read32(content, offset)
    #read image
    size_image = rows*cols
    fmt_image = '>'+str(size_image*num_images)+'B'
    buf = struct.unpack_from(fmt_image,content,offset)
    # data = np.frombuffer(buf, dtype=np.uint8)
    data = np.array(buf, dtype=np.uint8)

    data = data.reshape(num_images, rows*cols)

    return  data;

def read_labels_data(content):
    print("read labels data...")
    offset = 0
    magic, offset = _read32(content, offset)
    if magic != 2049:
        raise ValueError('Invaild magic number in mnist image file')

    num_items, offset = _read32(content, offset)

    fmt_items = ">"+str(num_items)+'B'
    buf = struct.unpack_from(fmt_items,content,offset)
    labels = np.array(buf, dtype=np.uint8)
    labels = labels.reshape(num_items, 1)

    labels_one_hot = np.zeros((num_items, DATAS_CLASS_NUM), dtype=np.uint8)
    for idx in range(0, num_items):
        labels_one_hot[idx][labels[idx]] = 1

    return labels_one_hot

def split_data(datas_train, labels_train, validation_rate = 0.2):

    data_shape = datas_train.shape
    validation_size = int(data_shape[0] * validation_rate)

    datas_validation = datas_train[:validation_size]
    labels_validation = labels_train[:validation_size]
    datas_train = datas_train[validation_size:]
    labels_train = labels_train[validation_size:]

    return datas_train, labels_train, datas_validation, labels_validation

def read_data(datas_path, datas_train_file_name, labels_train_file_name, datas_test_file_name, labels_test_file_name):

    with open(datas_path+datas_train_file_name, 'r') as fid_datas_train:
        datas_train_content = fid_datas_train.read()
        datas_train = read_images_data(datas_train_content)

    with open(datas_path+labels_train_file_name, 'r') as fid_labels_train:
        labels_train_content = fid_labels_train.read()
        labels_train = read_labels_data(labels_train_content)

    with open(datas_path+datas_test_file_name, 'r') as fid_datas_test:
        datas_test_content = fid_datas_test.read()
        datas_test = read_images_data(datas_test_content)

    with open(datas_path+labels_test_file_name, 'r') as fid_labels_test:
        labels_test_content = fid_labels_test.read()
        labels_test = read_labels_data(labels_test_content)

    return datas_train, labels_train, datas_test, labels_test

#----------------def the conv neural network------------------------------#

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_varible(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


START_LEARNING_RATE = 0.01
def CNN_train(datas_train, labels_train, datas_validation, labels_validation):
    datas_shape = datas_train.shape
    datas_train.astype(np.float32)
    labels_train.astype(np.float32)
    datas_validation.astype(np.float32)
    labels_validation.astype(np.float32)

    x = tf.placeholder(dtype=tf.float32, shape=[None, datas_shape[1]])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #first layer, 32 features
    w_conv1 = weight_varible([5, 5, 1, 32])
    b_conv1 = bias_varible([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    #max pooling
    h_pool1 = max_pool_2x2(h_conv1)

    #second layer, 64 features
    w_conv2 = weight_varible([5,5,32,64])
    b_conv2 = bias_varible([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    #max pooling
    h_pool2 = max_pool_2x2(h_conv2)

    #first connection layer
    #input 7x7x64, output 1024
    w_fc1 = weight_varible([7*7*64, 1024])
    b_fc1 = bias_varible([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #dropout, control the complexity of the model
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #map the 1024 features to 10 classes
    w_fc2 = weight_varible([1024, 10])
    b_fc2 = bias_varible([10])

    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    #train
    y_ = tf.placeholder(tf.float32, [None, 10])

    #learning rate
    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = START_LEARNING_RATE
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)

    #loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy,global_step=global_step)
    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_size = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_idx = 0
        for idx in range(0, 40000):
            batch_size = idx/40 + 50

            _, loss, prob = sess.run([train_step, cross_entropy, keep_prob], feed_dict={
                x:(datas_train[data_idx:data_idx+batch_size]), y_:(labels_train[data_idx:data_idx+batch_size]), keep_prob: 0.5})
            if data_idx>datas_shape[0]-batch_size-1:
                data_idx = data_idx+batch_size
            else:
                data_idx = 0

            if idx%50 == 0:
                accuracy_validation = accuracy.eval(feed_dict={x:(datas_validation[0:100]), y_:(labels_validation[0:100]), keep_prob:1.0})

                print('step: %10d, loss: %10g, validation accuracy: %10g'% (idx, loss, accuracy_validation))

    return




def CNN_test():

    return

if __name__ == '__main__':


    datas_train, labels_train, datas_test, labels_test = read_data(DATAS_PATH,
                                                                   DATAS_TRAIN_FILE_NAME,
                                                                   LABELS_TRAIN_FILE_NAME,
                                                                   DATAS_TEST_FILE_NAME,
                                                                   LABELS_TEST_FILE_NAME)

    datas_train, labels_train, datas_validation, labels_validation = split_data(datas_train, labels_train, 0.2)

    CNN_train(datas_train, labels_train, datas_validation, labels_validation)

    print('end')
