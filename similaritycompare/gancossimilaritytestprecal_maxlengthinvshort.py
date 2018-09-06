import csv
import pickle
import numpy as np
import math

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

source2sinkdatafile = open('source2sinkdata.obj', 'r')
source2sinkdata = pickle.load(source2sinkdatafile)
print('source2sinkdata loaded')



def similaritybetweeneachsourcewheresinkexist_comparetosource(sourceinput, sinkinput, data, sourceorder):
    set_sourceinfo = set(data[sourceorder.index(sourceinput)])
    orderlist = []
    sourcefollowedbysink = 0
    for source in range(len(data)):
        if (sinkinput == sourceorder[source]):
            if (sourceinput in data[source]):
                sourcefollowedbysink = 1
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
    if (length_orderlist > 0):
        term = 1
        listforgan.append(term)
        for j in range(4):
            term = term * orderlist[0]
            listforgan.append(term)
    else:
        term = 1
        listforgan.append(term)
        for j in range(4):
            listforgan.append(0)
    term = 1
    for j in range(2):
        term = term * length_orderlist
        listforgan.append(term)
    if (sourcefollowedbysink == 1):
        listforgan.append(1)
    else :
        listforgan.append(0)
    return listforgan


#print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1272125, source2sinkdata, sourcedata))
#print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1300190, source2sinkdata, sourcedata))


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


testsimlist = []
for i in range(len(testx)):
    print(i)
    testsimlist.append(similaritybetweeneachsourcewheresinkexist_comparetosource(testx[i][0], testx[i][1], source2sinkdata, sourcedata))



testsimlistfile = open('testsimlistfile.obj', 'w')
pickle.dump(testsimlist, testsimlistfile)




