import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
import HtmlTestRunner
import random


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

from RHML.Ensembles.Baggers import ClassificationBagger,RegressionBagger
from RHML.DTrees.ClassificationTree import RHMLClassificationDecisionTree
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
from RHML.Ensembles.Boosters import BaseGradBooster, RegressionGradBooster
from RHML.Ensembles.Boosters import AdaBooster

class TestBooster(unittest.TestCase):
    def test_gradientBooster(self):
        # first create some data (regression)
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
        
        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

        myBooster = RegressionGradBooster.RHMLRegressionGradBooster(50,X_train,y_train,**{"min_split":25,"max_depth":6}) 

        predictions= myBooster.predict(X_train)

        training_results = myBooster.test(X_train,y_train)
        print(f"Training data :")
        print(f"MSE: {training_results.score[1]}")
        print(f"R2: {training_results.score[0]}")

        test_results = myBooster.test(X_test,y_test)
        print(f"Test data :")
        print(f"MSE: {test_results.score[1]}")
        print(f"R2: {test_results.score[0]}")

        featureImp = myBooster.calcFeatureImportance()

        # check feature importance agasint canned output
        canned_fimp = np.array([0.11479883, 0.88520117])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # check training metrics
        canned_training_MSE = 3.2888038948508043
        canned_training_R2 = 0.9994027124744469
        self.assertAlmostEqual(canned_training_MSE,training_results.score[1])
        self.assertAlmostEqual(canned_training_R2,training_results.score[0])

        # check test metrics
        canned_test_MSE = 72.91115131703847
        canned_test_R2 = 0.985789297644661
        self.assertAlmostEqual(canned_test_MSE,test_results.score[1])
        self.assertAlmostEqual(canned_test_R2,test_results.score[0])

    def test_adaBooster(self):
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

        training_results = myBooster.test(X_train,y_train).score
        print(f"Training data :")
        print(f"Score: {training_results}")

        test_results = myBooster.test(X_test,y_test).score
        print(f"Test data :")
        print(f"Score: {test_results}")

        featureImp = myBooster.calcFeatureImportance()

        print(f"feature imp1 : {featureImp}")

        canned_training_score = 0.8955223880597015
        canned_testing_score = 0.8
        self.assertAlmostEqual(canned_training_score,training_results[0])
        self.assertAlmostEqual(canned_testing_score,test_results[0])

        # check feature importance agasint canned output
        canned_fimp1 = np.array([0.0745449 ,0.04382876, 0., 0., 0. ,0.21775478 ,0.12923428, 0.04196824 ,0. ,0.0566889, 0. ,0.04841167, 0.03942155 ,0., 0.09926057 ,0. ,0. ,0.05282958 ,0.05583084 ,0.14022592] )
        self.assertAlmostEqual(np.sum(canned_fimp1-featureImp),0)




if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner())