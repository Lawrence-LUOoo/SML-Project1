import tensorflow as tf


from random import randint

#rand = [randint(0, 5000000) for p in range(0, 10)]
#print(rand)
fileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/train.txt'
x = []
y = []
orifile = open(fileName, 'r')
for line in orifile.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    linelist = [int(i) for i in linelist]
    for i in range(len(linelist)):
        if (i > 0):
            x.append([linelist[0], linelist[i]])
            y.append(1) #1 means (origin, des) exist
    for j in range(10): #
        notexisttry = randint(0, 5000000)
        if (notexisttry not in linelist[1:]):
            x.append([linelist[0], linelist[i]])
            y.append(1) #1 means (origin, des) not exist

batch_size = 100000
#how many batches
n_batch = len(x) // batch_size


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



#Create two place holders
x = tf.placeholder(tf.float32, [None,2])
y = tf.placeholder(tf.float32, [None,2])

#define a simple NN
W = tf.Variable(tf.random_normal([2,2]))
b = tf.Variable(tf.zeros([2]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

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
            sess.run(train_step, feed_dict={x:x, y:y})
            predictionresult = sess.run(prediction, feed_dict={testx})
            print(predictionresult)
