# SML-Project1

### Object file

similaritycompare/outputdataid.py generate 5 python object files: 

- sourcedatalength.obj is the number of sources in train dataset 

- sinkdatalength.obj is the number of distinct sinks in train dataset 

- sourcedataidlist.obj is a hash table for source IDs in train dataset 

- sinkdataidlist.obj is a hash table for sink IDs in train dataset 

- source2sinkdata.obj read each row of sinks into list, one element in the list is one row, each row's correspond source is same as the source order in sourcedataidlist.obj 




source2sinktwodimensiontraindata.py generate one python object files: 

- twodimentiondatafile.obj is a list of edges in the train dataset 




source2sinksuffledata.py generate one python object files:

- shuffledtwodimentiondatafile.obj is a list which is generated by shuffle the elements in twodimentiondatafile.obj



### Similarity

- maxsimilarity.py uses the baseline similarity to generate predictions

- maxcossimilarity.py uses the cossimilarity to generate predictions

- maxcossimilarityasprobability.py uses the conditional probability to generate predictions


### GAN

gancossimilaritytestprecal.py

Transform the test datasets as a vector of function of each edges highest cossimilarity c1, second highest cossimilarity c2, third highest cossimilarity c3, fourth highest cossimilarity c4.

[1, c1, c1^2, c1^3, c1^4, 1, c2, c2^2, c2^3, c2^4, 1, c3, c3^2, c3^3, c3^4, 1, c4, c4^2, c4^3, c4^4]

Save to testsimlistfile.obj file





similaritycomparewithgan.py

Seem [1, c1, c1^2, c1^3, c1^4, 1, c2, c2^2, c2^3, c2^4, 1, c3, c3^2, c3^3, c3^4, 1, c4, c4^2, c4^3, c4^4] as predictor, use traditional GAN to predict the test file




gancossimilaritytestprecal_maxlengthinvshort.py

Transform the test datasets as a vector of function of each edges highest cossimilarity c1, the frequency of the sink f, whether source exists in sink's sink list in train set W

[1, c1, c1^2, c1^3, c1^4, 1/f, 1/f^2, w]

Save to testsimlistfile.obj file




ganmaxlengthinvcontrolgenerate.py

Seem [1, c1, c1^2, c1^3, c1^4, 1/f, 1/f^2, w] as predictor, use traditional GAN to predict the test file



