import csv
import pickle
import numpy as np


twodimentiondatafile = open('twodimentiondatafile.obj', 'r')
twodimentiondata = pickle.load(twodimentiondatafile)
print(twodimentiondata[:10])

idx = np.arange(0 , len(twodimentiondata))
np.random.shuffle(idx)

data_shuffle = [twodimentiondata[i] for i in idx]

print(data_shuffle[:10])
shuffledtwodimentiondatafile = open('shuffledtwodimentiondatafile.obj', 'w')
pickle.dump(data_shuffle, shuffledtwodimentiondatafile)

