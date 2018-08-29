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
            #print(sourceorder[source])
            #print('same sink')
            #print(set_sourceinfo & iter_sourceinfo_set)
            #print('number of same sinks')
            #print(len(set_sourceinfo & iter_sourceinfo_set))
            #print('number of set_sourceinfo sinks')
            #print(len(set_sourceinfo))
            #print('number of iter_sourceinfo_set sinks')
            #print(len(iter_sourceinfo_set))
            #return(float(len(set_sourceinfo & iter_sourceinfo_set))/max(len(set_sourceinfo), len(iter_sourceinfo_set)))
            orderlist.append(float(len(set_sourceinfo & iter_sourceinfo_set))/max(len(set_sourceinfo), len(iter_sourceinfo_set)))
    orderlist.sort(reverse = True)
    orderlist = filter(lambda a: a != 0, orderlist)
    if (len(orderlist) > 0):
        return (orderlist[0])
    else:
        return (0)

print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1272125, source2sinkdata, sourcedata))
print(similaritybetweeneachsourcewheresinkexist_comparetosource(4066935, 1300190, source2sinkdata, sourcedata))


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

with open('maxsimilarity.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Id', 'Prediction'])
    for i in range(len(testx)):
        print(i)
        filewriter.writerow([str(i+1), similaritybetweeneachsourcewheresinkexist_comparetosource(testx[i][0], testx[i][1], source2sinkdata, sourcedata)])





