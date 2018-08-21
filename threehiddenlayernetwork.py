import tensorflow as tf


from random import randint
import numpy as np
import csv


def next_batch(num, data, labels):
    '''
        Return a total of `num` random samples and labels.
        '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)





fileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/train.txt'
xdata = []
ydata = []
orifile = open(fileName, 'r')
for line in orifile.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    linelist = [int(i) for i in linelist]
    for i in range(len(linelist)):
        if (i > 0):
            xdata.append([linelist[0], linelist[i]])
            ydata.append(1) #1 means (origin, des) exist
    for j in range(10): #
        notexisttry = randint(0, 5000000)
        if (notexisttry not in linelist[1:]):
            xdata.append([linelist[0], linelist[i]])
            ydata.append(0) #0 means (origin, des) not exist

batch_size = 1000
#how many batches
n_batch = len(xdata) // batch_size
print(n_batch)
xdata = np.asarray(xdata, dtype=np.float32)
ydata = np.asarray(ydata, dtype=np.float32)
print(xdata)
testfileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/test-public.txt'
testx = []
testfile = open(testfileName, 'r')
m=0
for line in testfile.readlines()[1:]:
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    linelist[1:] = [int(i) for i in linelist[1:]]
    testx.append(linelist[1:])
testx = np.asarray(testx, dtype=np.float32)


#Create two place holders
x = tf.placeholder(tf.float32, [None,2])
y = tf.placeholder(tf.float32, [None,2])



#initialize weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#initialize bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)



#initialize the 1st layer
W_1 = weight_variable([2,10])
b1 = bias_variable([10])
s1 = tf.nn.softmax(tf.matmul(x,W_1)+b1)



#initialize the 2nd layer
W_2 = weight_variable([10,10])
b2 = bias_variable([10])
s2 = tf.nn.softmax(tf.matmul(s1,W_2)+b2)


#initialize the 3rd layer
W_3 = weight_variable([10, 2])
b3 = bias_variable([2])
eta = tf.matmul(s2,W_3)+b3
prediction = tf.nn.softmax(eta)










#quadratic cost function
loss = tf.reduce_mean(tf.square(y-prediction))
#define a gradient descent to train the Optimizer
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)

init = tf.global_variables_initializer()

#boolean list
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)) #argmax return the position of maximum
#
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(22):
        for batch in range(n_batch):
            batch_xdata, batch_ydata = next_batch(n_batch, xdata, ydata)
            sess.run(train_step, feed_dict={x:batch_xdata, y:batch_xdata})
            batch_xdata, batch_ydata = next_batch(20, xdata, ydata)
            acc = sess.run(accuracy, feed_dict={x:batch_xdata, y:batch_xdata})
            print("Iter " + str(epoch) + ",Training Accuracy " + str(acc))
        predictionresult = sess.run(prediction, feed_dict={x:testx})
        with open('persons.csv', 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Id', 'Prediction'])
            print(predictionresult.shape)
            for i in range(predictionresult.shape[0]):
                filewriter.writerow([str(i+1), predictionresult[i, 1]])
        print(predictionresult)
