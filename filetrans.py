from random import randint

#rand = [randint(0, 5000000) for p in range(0, 10)]
#print(rand)
fileName = '/Users/chen/Desktop/study/COMP90051 Statistical Machine Learning/ass/train.txt'
x = []
y = []
orifile = open(fileName, 'r')
m=0
for line in orifile.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    linelist = line.split('\t')
        #if (m == 0):
        #print (linelist)
        #m = 1
    for i in range(len(linelist)):
        #if (linelist[0] == '2184483'):
        #    if ('1300190' in linelist):
        #        print(linelist)
        if (i > 0):
            x.append([linelist[0], linelist[i]])
            y.append(1) #1 means (origin, des) exist
    for j in range(10): #
       notexisttry = randint(0, 5000000)
       if (notexisttry not in linelist[1:]):
           x.append([linelist[0], linelist[i]])
           y.append(1) #1 means (origin, des) not exist
#print x
#   print y
#for i in line.split(" "):

#row[-1].append(i)
