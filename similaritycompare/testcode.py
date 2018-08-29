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
    print(orderlist)
    if (len(orderlist) > 0):
        sumprob = 0
        find_fail = 1
        for prob in orderlist:
            sumprob = sumprob + find_fail * prob
            find_fail = 1 - sumprob
        
        return (sumprob)
    else:
        return (0)

print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1272125, source2sinkdata, sourcedata))
print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1300190, source2sinkdata, sourcedata))
print('test id 2')
print(similaritybetweeneachsourcewheresinkexist_comparetosource(3151356, 1452193, source2sinkdata, sourcedata))
print('test id 45')
print(similaritybetweeneachsourcewheresinkexist_comparetosource(195757, 4324049, source2sinkdata, sourcedata))


