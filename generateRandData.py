import random
import csv

## read and parser data in file
def parserData(filePath):
    dict = {}
    sourceList =[]
    with open(filePath,'r') as file:
        for line in file:
            ls = line.split()
            source = ls.pop(0)
            sourceList.append(source) ## store sources in list
            dict[source]= ls          ## assign list ot dict
    return dict, sourceList

## Generate 7 digits number
def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end)


## Pick the random source and sink -----> real
def realSourceSink(dict, sourceList): 
    source = random.choice(sourceList)
    try:
        sink = random.choice(dict[source])
        return [source,sink,1]
    except IndexError:
        return None
    

def fakeSourceSink(sourceList,digits):
    source = random.choice(sourceList) 
    sink = random_with_N_digits(digits)
    return [source,sink,0]
    
## Generate the list of real and fake data
def listData (dict, sourceList, maxNumber,digits):
    dataList = []
    for i in range(maxNumber):
        threshold = bool(random.getrandbits(1))
        if (threshold):
            data=realSourceSink(dict,sourceList)
            while(data is None): data = realSourceSink(dict,sourceList)
            dataList.append(data)
        else:
            dataList.append(fakeSourceSink(sourceList,digits))
    return dataList


## Write out list of data to csv 
def writeCsvResult(dataList):
    with open('dataResults.csv','w') as resultFile:
        writer = csv.writer(resultFile)
        writer.writerow(["Id", "Source", "Sink", "True or False"])
        for i in range(len(dataList)):
                elem = list(dataList[i])
                elem.insert(0,i+1)
                writer.writerow(elem)
            


def  writeCsvTest (dataList):
    with open('dataTest.csv','w') as testFile:
        writer = csv.writer(testFile)
        writer.writerow(["Id", "Source", "Sink"])
        for i in range(len(dataList)):
                elem = list(dataList[i])
                elem.insert(0,i+1)
                del elem[-1]
                writer.writerow(elem)


def main():
    ## initialize dictionary and list for sources
    dict = {}
    sourceList=[]
    filePath = "../Data/train.txt"

    ## Data create
    dict, sourceList = parserData(filePath)


    ## Create data set 
    dataList= listData (dict,sourceList,10,7)

    ## Generate data combination
    writeCsvResult(dataList)
    writeCsvTest(dataList)
    


if __name__== "__main__":
    main()