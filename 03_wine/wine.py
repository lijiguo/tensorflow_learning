import tensorflow as tf
import numpy as np

FEATURE_LEN = 13
CLASS_NUM = 3
def read_data():
    filename = 'wine.txt'
    with open(filename) as fid:
        data_str = fid.read()

    data_str_line = data_str.split('\r\n')

    num_line = len(data_str_line)

    datas = np.zeros((num_line, FEATURE_LEN))

    labels = np.zeros((num_line, CLASS_NUM),dtype=np.uint8)
    for idx in range(0,num_line):
        data_str_line_char = data_str_line[idx].split(',');
        #read the label
        label = int(data_str_line_char[0])
        if label>=1 and label<=3:
            labels[idx][label-1] = 1

        #read the feature vector
        for feature_idx in range(0,FEATURE_LEN):
            datas[idx][feature_idx] = float(data_str_line_char[feature_idx-1])

    #normalize
    datas_min = np.amin(datas, axis=0)
    datas_max = np.amax(datas, axis=0)
    datas_range = datas_max - datas_min
    for idx in range(0,FEATURE_LEN):
        for data_idx in range(0,num_line):
            datas[data_idx][idx] = (datas[data_idx][idx] - datas_min[idx])/datas_range[idx]

    return datas,labels;

'''
    :param datas: an observer each row
    :param labels: ont hot
    :return: datas_train, label_train, datas_test, label_test
'''
def split_train_test(datas, labels):

    shape_datas = datas.shape;
    shape_label = labels.shape;

    if shape_datas[0] != shape_label[0]:
        print('data error!')
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0);

    num_class = shape_label[1]

    total_num_each_class = np.zeros(num_class, dtype=np.int32)

    #num of every class
    for idx in range(0,shape_datas[0]):
        label_idx = np.nonzero(labels[idx])[0]
        if label_idx <= num_class:
            total_num_each_class[label_idx] = total_num_each_class[label_idx]+1;

    #random the data, ignore this step
    #allocate the data
    train_num_each_class = (total_num_each_class * 0.8).astype(np.int32)

    size_train_data = np.sum(train_num_each_class)
    size_test_data = np.sum(total_num_each_class - train_num_each_class)
    datas_train = np.zeros((size_train_data, shape_datas[1]))
    labels_train = np.zeros((size_train_data, shape_label[1]))
    datas_test = np.zeros((size_test_data, shape_datas[1]))
    labels_test = np.zeros((size_test_data, shape_label[1]))

    train_num_cur = np.zeros(num_class, dtype=np.int32)
    test_num_cur = np.zeros(num_class, dtype=np.int32)
    for idx in range(0,shape_label[0]):
        label_idx = np.nonzero(labels[idx])[0]
        if label_idx > num_class:
            print("data error")

        if train_num_cur[label_idx]<train_num_each_class[label_idx]:
            datas_train[np.sum(train_num_cur)] = datas[idx];
            labels_train[np.sum(train_num_cur)] = labels[idx];
            train_num_cur[label_idx] = train_num_cur[label_idx] + 1;
        else:
            datas_test[np.sum(test_num_cur)] = datas[idx];
            labels_test[np.sum(test_num_cur)] = labels[idx];
            test_num_cur[label_idx] = test_num_cur[label_idx] + 1;

    return datas_train, labels_train, datas_test, labels_test



HIDDEN_LAYER_NUM = 13
LEARNING_RATE = 0.01
def NN_train(datas_train, labels_train, datas_test, labels_test, fid):
    shape_data = datas_train.shape
    shape_label = labels_train.shape
    num_class = shape_label[1]
    len_feature = shape_data[1]
    #check the data
    if shape_data[0] != shape_label[0]:
        print("NN_train: data error")


    #tensor map
    x = tf.placeholder(tf.float32, [None, len_feature])
    w1 = tf.Variable(tf.zeros([len_feature, HIDDEN_LAYER_NUM]))
    b1 = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM]))

    x2 = tf.nn.sigmoid(tf.matmul(x,w1) + b1)

    w2 = tf.Variable(tf.zeros([HIDDEN_LAYER_NUM, num_class]))
    b2 = tf.Variable(tf.zeros([num_class]))
    x3 = tf.nn.sigmoid(tf.matmul(x2, w2) + b2)
    #
    y = tf.nn.softmax(x3)

    #learning rate

    #how to train
    y_ = tf.placeholder(tf.float32, [None, num_class])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-6,1.0)), reduction_indices=[1]))

    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = LEARNING_RATE
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1000, 0.9, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

    #evaluate the model
    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #run
    init = tf.initialize_all_variables()
    batch_size = 10
    with tf.Session() as sess:
        sess.run(init)
        data_idx = 0;
        for idx in range(20000):
            _,loss,learning_rate_temp = sess.run([train_step, cross_entropy, learning_rate], feed_dict={x:datas_train[data_idx:data_idx+batch_size], y_:labels_train[data_idx:data_idx+batch_size]})
            print>>fid, "%10d\taccuracy:%f\tloss:%f\tlearning rate:%f"%(idx, sess.run(accuracy, feed_dict={x:datas_test, y_:labels_test}),loss, learning_rate_temp)

            if data_idx<shape_data[0]-batch_size-1:
                data_idx = data_idx+1
            else:
                data_idx = 0

            if idx % 500 == 0:
                print "%10d\taccuracy:%f\tloss:%f\tlearning rate:%f"%(idx, sess.run(accuracy, feed_dict={x:datas_test, y_:labels_test}),loss, learning_rate_temp)



import matplotlib.pyplot as plt
import string
def plot_log_data(filename):
    fid = open(filename,'r')
    accuracy = []
    loss = []
    learning_rate = []

    while True:
        line = fid.readline()
        if line == '':
            break;

        line_split = line.split('\t')
        if len(line_split) != 4:
            print('plot_log_data: data error')

        idx = string.atoi(line_split[0])
        pair_split = line_split[1].split(':')
        accuracy_temp = string.atof(pair_split[1])
        pair_split = line_split[2].split(':')
        loss_temp = string.atof(pair_split[1])
        pair_split = line_split[3].split(':')
        learning_rate_temp = string.atof(pair_split[1])

        accuracy.append(accuracy_temp)
        loss.append(loss_temp)
        learning_rate.append(learning_rate_temp)

    fid.close()
    #plot

    x = np.arange(0,len(accuracy),1)

    ax = plt.subplot(111)
    line1 = ax.plot(x, accuracy, color='red', label = 'accuracy')
    line2 = ax.plot(x, loss, color='blue', label = 'loss')
    ax.legend(loc = 'upper right')
    plt.show()




if __name__ == '__main__':

    file_name = 'log.txt'
    # fid = open(file_name,'w')
    # print('read data...')
    # datas,labels = read_data()
    # print('split data...')
    # datas_train, labels_train, datas_test, labels_test = split_train_test(datas,labels)
    # print('train the neural network...')
    # NN_train(datas_train, labels_train, datas_test, labels_test, fid)
    # print 'end, log file is %s'%file_name
    # fid.close();

    plot_log_data(file_name)


