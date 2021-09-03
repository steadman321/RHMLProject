
# START PATH MAGIC!!!
# Note - this magic path jumping through hoops only required as my test suite is at same level as my code 
# For other users, they would need to either have RHML dir in the PYTHONPATH or system path, or have it in same dir as their other code 
# Code below adds 'RHML' to the runtime path
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# END MAGIC

from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random



from RHML.Ensembles.Boosters import BaseGradBooster, RegressionGradBooster
from RHML.Ensembles.Boosters import AdaBooster
from RHML.DTrees.ClassificationTree import RHMLClassificationDecisionTree
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
from RHML.Utils.Plots import plotFeatureImportance


def simpleBaseBoosterRegressionTest():
    
    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBooster = BaseGradBooster.RHMLBooster(50,X_train,y_train,**{"min_split":25,"max_depth":6})

    # predictions = myBagger.predict(X_test)
    # print(f"size eof test data samples {len(X_test)}")
    # print(f"sixe of final predictions {len(predictions)}")
    # print(f"final predictions are : {predictions}")    

    predictions= myBooster.predict(X_train)
    print(f"my new prediciotns are : {predictions}")

    training_results = myBooster.test(X_train,y_train)
    print(f"Training data :")
    print(f"MSE: {training_results.score[1]}")
    print(f"R2: {training_results.score[0]}")

    test_results = myBooster.test(X_test,y_test)
    print(f"Test data :")
    print(f"MSE: {test_results.score[1]}")
    print(f"R2: {test_results.score[0]}")


def regressionBoosterTest():
    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBooster = RegressionGradBooster.RHMLRegressionGradBooster(50,X_train,y_train,**{"min_split":25,"max_depth":6})

    # predictions = myBagger.predict(X_test)
    # print(f"size eof test data samples {len(X_test)}")
    # print(f"sixe of final predictions {len(predictions)}")
    # print(f"final predictions are : {predictions}")    

    predictions= myBooster.predict(X_train)
    print(f"my new prediciotns are : {predictions}")

    training_results = myBooster.test(X_train,y_train)
    print(f"Training data :")
    print(f"MSE: {training_results.score[1]}")
    print(f"R2: {training_results.score[0]}")

    test_results = myBooster.test(X_test,y_test)
    print(f"Test data :")
    print(f"MSE: {test_results.score[1]}")
    print(f"R2: {test_results.score[0]}")

    featureImp = myBooster.calcFeatureImportance()
    plotFeatureImportance([featureImp,featureImp])

def adaBoosterTestOld():
    # first create some data (classification)
    # X, y = make_blobs(n_samples=1000,n_features=2, centers=2, random_state=101)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)


    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

    myBooster = AdaBooster.RHMLAdaBooster(15,X_train,y_train,**{"min_split":5,"max_depth":3})

    predictions= myBooster.predict(X_train)
    print(f"my new prediciotns are : {predictions}")
    print(f"And my actuals are {y_train}")
    correct = np.sum(predictions==y_train)
    print(f"number correct = {correct} out of {len(predictions)}, as a percent : {100*(correct/len(predictions))}")

    training_results = myBooster.test(X_train,y_train)
    print(f"Training data :")
    print(f"Score: {training_results}")

    test_results = myBooster.test(X_test,y_test)
    print(f"Test data :")
    print(f"Score: {test_results}")

def adaBoosterTest():
    random.seed(101)
    # first create some data (classification)
    # X, y = make_blobs(n_samples=1000,n_features=2, centers=2, random_state=101)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)


    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

    myBooster = AdaBooster.RHMLAdaBooster(15,X_train,y_train,**{"min_split":1,"max_depth":2})

    predictions= myBooster.predict(X_train)
    print(f"my new prediciotns are : {predictions}")
    print(f"And my actuals are {y_train}")
    correct = np.sum(predictions==y_train)
    print(f"number correct = {correct} out of {len(predictions)}, as a percent : {100*(correct/len(predictions))}")

    training_results = myBooster.test(X_train,y_train)
    print(f"Training data :")
    print(f"Score: {training_results}")

    test_results = myBooster.test(X_test,y_test)
    print(f"Test data :")
    print(f"Score: {test_results}")

    featureImp = myBooster.calcFeatureImportance()
    # print("goingt ograph it")
    plotFeatureImportance([featureImp,featureImp2])


def testStuff():
    arr2D = np.array([
                        [1, 12, 13,2],
                        [14, 15, 16,45],
                        [17, 15, 11,12]
                    ])
    print(arr2D)

    # find max per col
    maxInColumns = np.amax(arr2D, axis=0)
    print(f"Max in cols : {maxInColumns}")

    # Get the indices of maximum element in numpy array
    result = np.where(arr2D == np.amax(arr2D, axis=0))
    print(f"max per col : index = {result}")

    maxPerCol = np.argmax(arr2D,axis=0)
    print(f"maxperCol {maxPerCol}")


# simpleBaseBoosterRegressionTest()
# regressionBoosterTest()
# adaBoosterTestOld()
adaBoosterTest()
# testStuff()
