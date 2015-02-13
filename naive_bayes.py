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


num_train = 45000
num_test = 5000
# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def main():
    # Bag of words for training data
    file_train = "../train_emails_bag_of_words_200.dat"

    # Bag of words for test data
    file_test = "../test_emails_bag_of_words_0.dat"
    train = read_bagofwords_dat(file_train, num_train)
    test = read_bagofwords_dat(file_test, num_test)

    train_target = []
    for i in range(0, num_train):
        if i < num_train/2:
            train_target.append("notspam")
        else:
            train_target.append("spam")

    test_target = []
    for i in range(0, num_test):
        if i < num_test/2:
            test_target.append("notspam")
        else:
            test_target.append("spam")

    
    gnb = GaussianNB()
    model = gnb.fit(train, train_target)
    print "Finished training\n"
    y_pred = model.predict(test)

        
    print("Number of mislabeled test points out of a total %d points : %d" 
          % (len(y_pred),(test_target != y_pred).sum()))

main()
