import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
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

class TestClassificationTrees(unittest.TestCase):
    def test_simpleClassificsatoinTree(self):
        # first create some data (classification)
        x_train, y_train = make_blobs(n_samples=100,n_features=2, centers=3, random_state=101)
        
        # now make up some test observations
        testObservations = []
        testObservations.append(np.array([-10,-6])) #class1
        testObservations.append(np.array([4,7.5]))  #class2
        testObservations.append(np.array([0,1]))    #class0

        # now try classification tree
        from RHML.DTrees import ClassificationTree as CTree 
        ctree = CTree.RHMLClassificationDecisionTree(x_train,y_train)
        ctree.describe()
        featureImp = ctree.calcFeatureImportance()

        results = ctree.predict(testObservations)

        # also work out probs for each class for each test observation 
        class0, probs0 = ctree.calcClassProbabilities(testObservations[0])
        class1, probs1 = ctree.calcClassProbabilities(testObservations[1])
        class2, probs2 = ctree.calcClassProbabilities(testObservations[2])

        # canned output 
        numberOfNodes = len(ctree.nodes)
        self.assertEqual(numberOfNodes,5)

        # check feature importance agasint canned output
        canned_fimp = np.array([0.4975617 , 0.5024383])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # check predictions against canned predictions
        canned_predictions = np.array([1,2,0])
        self.assertAlmostEqual(np.sum(canned_predictions-results),0)

        # check class probs against canned predictions
        canned_class_probs = np.array([1,1,1])
        actual_class_probs  = np.array([probs0,probs1,probs2])
        self.assertAlmostEqual(np.sum(canned_class_probs-actual_class_probs),0)

    def test_performanceClassificationTree(self):
        # first create some data (classification)
        X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 
    
        # now create (and fit) classification tree
        from RHML.DTrees import ClassificationTree as CTree 
        ctree = CTree.RHMLClassificationDecisionTree(X_train,y_train,**{"min_split":150})

        ctree.describe()
        featureImp = ctree.calcFeatureImportance()

        # Lets get some stats for the training data first! 
        training_results = ctree.test(X_train,y_train)
        training_cmatrix = training_results.score[1]

        
        # now do for test data 
        test_results = ctree.test(X_test,y_test)
        test_cmatrix = test_results.score[1]

        
        # Check for truth ..... 
         
        # nodes should be 5
        numberOfNodes = len(ctree.nodes)
        self.assertEqual(numberOfNodes,5)

        # check feature importance agasint canned output
        canned_fimp = np.array([0.49579135 , 0.50420865])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # Training data : measure1: 
        canned_training_meansure1 = 1.0
        self.assertAlmostEqual(canned_training_meansure1,training_results.score[0])

        # Training data : confusion matrix
        canned_training_cmatrix = np.array([[226., 0., 0.] ,[ 0., 217., 0.] ,[ 0., 0. ,227.]])
        tester = training_cmatrix.getConfusionMatrix()
        np.testing.assert_array_equal(canned_training_cmatrix,training_cmatrix.getConfusionMatrix())
        
        # Test data : measure1: 
        canned_test_meansure1 = 0.996969696969697
        self.assertAlmostEqual(canned_test_meansure1,test_results.score[0])

        # Test data : confusion matrix
        canned_test_cmatrix = np.array([[108. ,0. ,0.] ,[ 0., 116., 0.], [ 1., 0. ,105.]])
        np.testing.assert_array_equal(canned_test_cmatrix,test_cmatrix.getConfusionMatrix())



if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner())