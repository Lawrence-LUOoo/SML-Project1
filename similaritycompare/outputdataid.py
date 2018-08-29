import pickle
fileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/small.txt'

sourcedata = []
sinkdata = []
source2sinkdata = []
orifile = open(fileName, 'r')
for line in orifile.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    linelist = [int(i) for i in linelist]
    sinkfromthissource = []
    for i in range(len(linelist)):
        if (i > 0):
            sinkdata.append(linelist[i])
            sinkfromthissource.append(linelist[i])
        
        else:
            sourcedata.append(linelist[i])
    source2sinkdata.append(sinkfromthissource)




#setsourcedata = set(sourcedata)
lensourcedatafile = open('sourcedatalength.obj', 'w')
setsourcelength = len(sourcedata)
pickle.dump(setsourcelength, lensourcedatafile)
print('setsourcelength saved')


setsinkdata = set(sinkdata)
lensinkdatafile = open('sinkdatalength.obj', 'w')
setsinkdatalength = len(setsinkdata)
pickle.dump(setsinkdatalength, lensinkdatafile)
print('setsinkdatalength saved')


sourcedataidlist = open('sourcedataidlist.obj', 'w')
#setsourcedatabacktolist = list(setsourcedata)
pickle.dump(sourcedata, sourcedataidlist)
print('setsourcedatabacktolist saved')

sinkdataidlist = open('sinkdataidlist.obj', 'w')
setsinkdatabacktolist = list(setsinkdata)
pickle.dump(setsinkdatabacktolist, sinkdataidlist)
print('setsinkdatabacktolist saved')


source2sinkdatafile = open('source2sinkdata.obj', 'w')
pickle.dump(source2sinkdata, source2sinkdatafile)
print('source2sinkdata saved')





