testfileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/test-public.txt'
testx = []
testfile = open(testfileName, 'r')
m=0
for line in testfile.readlines()[1:]:
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
    testx.append(linelist[1:])
    print (testx)

