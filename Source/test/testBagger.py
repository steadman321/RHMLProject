
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



from RHML.Ensembles.Baggers import BaseBagger,ClassificationBagger,RegressionBagger
from RHML.DTrees.ClassificationTree import RHMLClassificationDecisionTree
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
from RHML.Utils.Plots import plotFeatureImportance,plotProximityClassification,plotProximityRegresion


def simpleClassTest():
    random.seed(101)

    # first create some data (classification)
    X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)
    # X, y = make_blobs(n_samples=10,n_features=2, centers=3, random_state=101)

    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

    predictions = myBagger.predict(X_test)
    print(f"size eof test data samples {len(X_test)}")
    print(f"sixe of final predictions {len(predictions)}")
    print(f"final predictions are : {predictions}")
    print(f"OOB Score is : {myBagger.OOBScore}")

def simpleRegressionTest():
    random.seed(101)

    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

    predictions = myBagger.predict(X_test)
    print(f"size eof test data samples {len(X_test)}")
    print(f"sixe of final predictions {len(predictions)}")
    print(f"final predictions are : {predictions}")  
    print(f"OOB Score is : {myBagger.OOBScore}")  

def performanceClassTest():
    random.seed(101)

    # first create some data (classification)
    X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,
        X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

    # predictions = myBagger.predict(X_test)
    # print(f"size eof test data samples {len(X_test)}")
    # print(f"sixe of final predictions {len(predictions)}")
    # print(f"final predictions are : {predictions}")
    classTrainingResults = myBagger.test(X_train,y_train)
    print()
    print(f"Training Results : ")
    print(f"measure1: {classTrainingResults.score[0]}")
    print(f"measure2: ")
    cmatrix = classTrainingResults.score[1]
    print(f"{cmatrix.getConfusionMatrix()}")
    # myBagger.calcFeatureImportance()
    print(f"OOB Score is : {myBagger.OOBScore}")


    classTestResults = myBagger.test(X_test,y_test)
    print()
    print(f"Test Results : ")
    print(f"measure1: {classTestResults.score[0]}")
    print(f"measure2: ")
    cmatrix = classTestResults.score[1]
    print(f"{cmatrix.getConfusionMatrix()}")
    featureImp = myBagger.calcFeatureImportance()
    plotFeatureImportance([featureImp])


def performanceRegressionTest():
    random.seed(101)

    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

    # We should know the OOB before run any test : 
    print(f"OOB Score is : {myBagger.OOBScore}") 
    
    regTrainingResults = myBagger.test(X_train,y_train)
    print()
    print(f"Training Results : ")
    print(f"MSE: {regTrainingResults.score[1]}")
    print(f"R2: {regTrainingResults.score[0]}")
    



    regTestResults = myBagger.test(X_test,y_test)
    print()
    print(f"Test Results : ")
    print(f"MSE: {regTestResults.score[1]}")
    print(f"R2: {regTestResults.score[0]}")
    
    featureImp = myBagger.calcFeatureImportance()
    plotFeatureImportance([featureImp])



def simpleClassTest_Proximity():
    random.seed(101)

    # first create some data (classification)
    X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


    myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4})

    proximityMatrix = myBagger.calcProximity(X_train)
    plotProximityClassification(proximityMatrix,y_train)

def simpleRegressionTest_Proximity():
    random.seed(101)

    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

    myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4})

    proximityMatrix = myBagger.calcProximity(X_train)
    plotProximityRegresion(proximityMatrix,y_train)



# simpleClassTest()
# simpleRegressionTest()
# performanceClassTest()
performanceRegressionTest()
# simpleClassTest_Proximity()
# simpleRegressionTest_Proximity()
