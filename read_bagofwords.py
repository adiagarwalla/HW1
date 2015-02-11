import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB

numofemails = 10000

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def main():
    file_dat = sys.argv[1]
    bagofwords = read_bagofwords_dat(file_dat)
    
    target = []
    # 0 = not_spam, 1 = spam 
    for i in range(0, numofemails):
        if i < numofemails/2:
            target.append(0)
        else:
            target.append(1)
    
    gnb = GaussianNB()
    model = gnb.fit(bagofwords, target)
    print "Finished training\n"
    y_pred = model.predict(bagofwords)
    
    print("Number of mislabeled points out of a total %d points : %d" 
          % (bagofwords.shape[0],(target != y_pred).sum()))

main()
