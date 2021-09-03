import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import HtmlTestRunner

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

class TestRegressionTrees(unittest.TestCase):
    def test_simpleRegressionTree(self):
        # first create some data (regression)
        x_train, y_train = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
        # now make up some test observations
        testObservations = []
        testObservations.append(np.array([0,-2.5])) #bottom
        testObservations.append(np.array([0,4])) #top 
        testObservations.append(np.array([0,0])) #mid
        testObservations.append(np.array([0,-2.5])) #bottom
        testObservations.append(np.array([0,4])) #top 
        testObservations.append(np.array([0,0])) #mid

        # now fit data to a regression tree
        from RHML.DTrees import RegressionTree as RTree 
        rtree = RTree.RHMLRegressionDecisionTree(x_train,y_train)
        rtree.describe()
        featureImp=rtree.calcFeatureImportance()

        # now get some predictions ......... 
        results = rtree.predict(testObservations)

        # Now assert the truth
        numberOfNodes = len(rtree.nodes)
        self.assertEqual(numberOfNodes,123)

        # check feature importance agasint canned output
        canned_fimp = np.array([0.07997798,0.92002202])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # check predictions against canned predictions
        canned_predictions = np.array([-149.46453317852558, 178.5382882375638, -6.335825793398959, -149.46453317852558, 178.5382882375638, -6.335825793398959])
        self.assertAlmostEqual(np.sum(canned_predictions-results),0)

    def test_performanceRegressionTree(self):
        # first create some data (regression)
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)

        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    
        # now fit data to a regression tree
        from RHML.DTrees import RegressionTree as RTree 
        rtree = RTree.RHMLRegressionDecisionTree(X_train,y_train,**{"min_split":25,"max_depth":9})
        rtree.describe()
        featureImp = rtree.calcFeatureImportance()

        # Lets get some stats for the training data first! 
        training_results = rtree.test(X_train,y_train)

        # now get some predictions 
        test_results = rtree.test(X_test,y_test)

        # # Now assert the truth
        numberOfNodes = len(rtree.nodes)
        self.assertEqual(numberOfNodes,87)

        # check feature importance agasint canned output
        canned_fimp = np.array([0.11780977,0.88219023])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # check training metrics
        canned_training_MSE = 155.75536126059274
        canned_training_R2 = 0.9717128970612625
        self.assertAlmostEqual(canned_training_MSE,training_results.score[1])
        self.assertAlmostEqual(canned_training_R2,training_results.score[0])

        # check test metrics
        canned_test_MSE = 353.64293441658504
        canned_test_R2 = 0.9310734450041751
        self.assertAlmostEqual(canned_test_MSE,test_results.score[1])
        self.assertAlmostEqual(canned_test_R2,test_results.score[0])




if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner())