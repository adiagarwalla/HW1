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
    file_train = "../train_emails_bag_of_words_200.dat"
    file_test = "../test_emails_bag_of_words_0.dat"
    train = read_bagofwords_dat(file_train)
    test = read_bagofwords_dat(file_test)

    train_target = []
    test_target = []
    for i in range(0, numofemails):
        if i < numofemails/2:
            train_target.append("spam")
            test_target.append("spam")
        else:
            train_target.append("notspam")
            test_target.append("notspam")
    
    gnb = GaussianNB()
    model = gnb.fit(train, train_target)
    print "Finished training\n"
    y_pred = model.predict(test)

        
    print("Number of mislabeled points out of a total %d points : %d" 
          % (numofemails,(test_target != y_pred).sum()))

main()
