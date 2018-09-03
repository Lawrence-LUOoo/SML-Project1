import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import math

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 200 
LR_G = 0.0001    # learning rate for generator
LR_D = 0.0001    # learning rate for discriminator
N_IDEAS = 10             # think of this as number of ideas for generating an art work (Generator)
number_of_useorders = 4*5  #number of high similarity used * (order of similarity + 1)


lensourcedatafile = open('sourcedatalength.obj', 'r')
setsourcelength = pickle.load(lensourcedatafile)
print(setsourcelength)

lensinkdatafile = open('sinkdatalength.obj', 'r')
setsinkdatalength = pickle.load(lensinkdatafile)
print(setsinkdatalength)

sourcedataidlist = open('sourcedataidlist.obj', 'r')
sourcedata = pickle.load(sourcedataidlist)
print('sourcedata loaded')

sinkdataidlist = open('sinkdataidlist.obj', 'r')
sinkdata = pickle.load(sinkdataidlist)
print('sinkdata loaded')

shuffledtwodimentiondatafile = open('shuffledtwodimentiondatafile.obj', 'r')
data_shuffle = pickle.load(shuffledtwodimentiondatafile)
print('source2sinkshuffleddata loaded')


testsimlistfile = open('testsimlistfile.obj', 'r')
testsimlist = pickle.load(testsimlistfile)
print('testsimlist loaded')

lendata_shuffle = len(data_shuffle)
print(lendata_shuffle)

def similaritybetweeneachsourcewheresinkexist_comparetosource_fortrain(sourceinput, sinkinput, data, sourceorder):
    set_sourceinfo = set(data[sourceorder.index(sourceinput)])
    orderlist = []
    for source in range(len(data)):
        if (sinkinput in data[source]):
            iter_sourceinfo_set = set(data[source])
            cos = 0
            if (len(set_sourceinfo & iter_sourceinfo_set) == 0):
                cos = 0
            else:
                cos = float(len(set_sourceinfo & iter_sourceinfo_set)) / math.sqrt(len(set_sourceinfo) * len(iter_sourceinfo_set))
            orderlist.append(cos)
    orderlist.sort(reverse = True)
    orderlist = filter(lambda a: a != 0, orderlist)
    listforgan = []
    length_orderlist = len(orderlist)
    k = 0
    for i in range(4):
        if (length_orderlist > k):
            while (orderlist[k] >= 1):
                k = k + 1
            term = 1
            listforgan.append(term)
            for j in range(4):
                term = term * orderlist[k]
                listforgan.append(term)
        else:
            term = 1
            listforgan.append(term)
            for j in range(4):
                listforgan.append(0)
        k = k + 1
    return listforgan


def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    return paintings


with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # random ideas (could from normal distribution)
    G_l1 = tf.layers.dense(G_in, 32, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, number_of_useorders, tf.nn.sigmoid)               # making a orderlist from these random ideas

with tf.variable_scope('Discriminator'):
    real_order = tf.placeholder(tf.float32, [None, number_of_useorders], name='real_in')   # receive order from the real data
    D_l0 = tf.layers.dense(real_order, 32, tf.nn.relu, name='l')
    prob_orderreal0 = tf.layers.dense(D_l0, 1, name='out')              # probability that the order is real
    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out, 32, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
    prob_orderreal1 = tf.layers.dense(D_l1, 1, name='out', reuse=True)  # probability that the order is real
theta_D = [D_l0, prob_orderreal0, D_l1, prob_orderreal1]
D_loss = -tf.reduce_mean(prob_orderreal0) + tf.reduce_mean(prob_orderreal1)
G_loss = tf.reduce_mean(-prob_orderreal1)

train_D = tf.train.RMSPropOptimizer(LR_D,0.9).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.RMSPropOptimizer(LR_G,0.9).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
prob_orderreal0 = tf.clip_by_value(prob_orderreal0, 0.0, 1.0)
prob_orderreal1 = tf.clip_by_value(prob_orderreal1, 0.0, 1.0)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver.restore(sess, 'wgantmp/model.ckpt')
print("train start")
#i = 44*BATCH_SIZE 
i = 0
for step in range(50000):
    print("epoch " + str(step+0))
    print("start from " + str(i))
    if (i > (lendata_shuffle - 3)):
        i = 0
    dataepoch = data_shuffle[i:(i+BATCH_SIZE)]
    i = i + BATCH_SIZE
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    batch_real = []
    for pair in dataepoch:
        batch_real.append(similaritybetweeneachsourcewheresinkexist_comparetosource_fortrain(pair[0], pair[1], data_shuffle, sourcedata))
    print("epoch start train")
    for iter in range(40):
        sess.run([train_D, train_G, prob_orderreal0, prob_orderreal1], {G_in: G_ideas, real_order: batch_real})
    print("test start")
    prediction_test = sess.run(prob_orderreal0, {real_order: testsimlist})
    csvname = str(step+0) + 'ganmethodcancel1intrain.csv'
    save_path = saver.save(sess, 'wgantmp/model.ckpt')
    with open(csvname, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Id', 'Prediction'])
        for n in range(prediction_test.shape[0]):
            filewriter.writerow([str(n+1), prediction_test[n, 0]])

