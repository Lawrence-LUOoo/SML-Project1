import pickle
fileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/small.txt'
twodimentiondata = []
orifile = open(fileName, 'r')
for line in orifile.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    linelist = [int(i) for i in linelist]
    for i in range(len(linelist)):
        if (i > 0):
            twodimentiondata.append([linelist[0], linelist[i]])



twodimentiondatafile = open('twodimentiondatafile.obj', 'w')
pickle.dump(twodimentiondata, twodimentiondatafile)

