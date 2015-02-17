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
from sklearn import naive_bayes, tree, svm, feature_selection
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

num_train = 45000
num_test = 5000
select_features = True
percentile = 10
freq_cutoff = 2.5
class_prior = [.2, .8]
rounds = 50

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def print_features(mask):
    vocab = []
    with open("trec07p_data/Test/train_emails_vocab_200.txt") as f:
        vocab = f.readlines()

    vocab = [x.strip('\n') for x in vocab]

    print "Retained vocabulary:"
    for i in range(0, num_test):
        if mask[i]:
            print vocab[i], 
    print "\n-----------------------"


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


    if select_features:
        selector = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=percentile)
        train = selector.fit_transform(train, train_target)
        test = selector.transform(test)
        
        mask = selector.get_support()
        #print_features(mask)
        print ("Finished doing %d percentile feature selection" % (percentile))

    classifiers = [
#        (svm.LinearSVC(), "SVML"),
#        (tree.DecisionTreeClassifier(), "Decision Tree"), 
#        (GaussianNB(), "Gaussian"), 
#        (MultinomialNB(1.0, False, class_prior), "Multinomial"), 
#        (BernoulliNB(1.0, freq_cutoff, False, class_prior), "Bernoulli"), 
#        (AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(max_depth=3), n_estimators = rounds), 
#        "Adaboost with %d rounds and max-depth 3 decision tree" % (rounds))
        ]
    for (classifier, name) in classifiers: 
        model = classifier.fit(train, train_target)
        y_pred = model.predict(test)
        FP = 0
        FN = 0
        TP = 0
        for i in range(0, num_test):
            if y_pred[i] == "spam" and test_target[i] == "notspam":
                FP+=1
            if y_pred[i] == "notspam" and test_target[i] == "spam":
                FN+=1
            if y_pred[i] == "spam" and test_target[i] == "spam":
                TP+=1

        print("%s: FP %d, FN %d, TP %d " % (name, FP, FN, TP))
'''
'''

main()
