import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
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

class TestBagger(unittest.TestCase):
    def test_simpleClassificationBagger(self):
        random.seed(101)

        # first create some data (classification)
        X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


        myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_oob":True})

        predictions = myBagger.predict(X_test)

        print(f"oobscore looks like this  {myBagger.OOBScore}")
        canned_OOB_score = (0.9983416252072969, )
        self.assertAlmostEqual(canned_OOB_score[0],myBagger.OOBScore[0])

        # predictions 
        canned_predictions = np.array([ 1, 1, 2, 0, 2, 2, 0, 1, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 0, 2, 
                                        0, 1, 0, 1, 0, 1, 1, 0, 2, 1, 2, 0, 2, 0, 0, 0, 1, 0, 1, 2, 2, 2, 1, 1, 
                                        1, 0, 1, 2, 1, 2, 1, 1, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 2, 2, 1, 
                                        0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 0, 1, 1, 0, 1, 1, 2, 2, 2, 
                                        2, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 2, 
                                        1, 0, 1, 2, 1, 1, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 
                                        1, 2, 0, 0, 0, 1, 2, 1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 1, 1, 
                                        0, 0, 2, 2, 1, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0, 1, 2, 0, 
                                        0, 0, 0, 2, 2, 1, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 1, 2, 2, 0, 1, 0, 2, 
                                        1, 1, 1, 0, 1, 2, 0, 0, 2, 1, 1, 2, 0, 2, 0, 1, 2, 2, 1, 2, 1, 1, 2, 1, 
                                        1, 2, 0, 2, 0, 2, 2, 0, 1, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 2, 
                                        2, 2, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 
                                        1, 0, 0, 1, 1, 0, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 
                                        0, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 2, 0, 1, 0, 0, 1, 2])

        np.testing.assert_array_equal(canned_predictions,predictions)

    def test_simpleRegressionBagger(self):
        random.seed(101)

        # first create some data (regression)
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
        
        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


        myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_oob":True})

        predictions = myBagger.predict(X_test)

        # NOTE : results here are : R2, MSE
        print(f"Bagger score looks like this {myBagger.OOBScore}")
        canned_OOB_score = (0.9139770942132152, 473.5602561733637)
        self.assertAlmostEqual(canned_OOB_score[0],myBagger.OOBScore[0])
        self.assertAlmostEqual(canned_OOB_score[1],myBagger.OOBScore[1])

        print(f"Predictions look like this : {predictions}")

        # canned_predictions = np.array([-56.694032704868064, 2.660273284119397, 64.88898127272921, 5.759918313758537, 64.88898127272921, 60.41884490473026, -21.423945954417384, 5.759918313758537, 90.8774369507012, 5.759918313758537, 5.759918313758537, 5.759918313758537, 36.89181577133981, 35.785405868173655, 9.677113743211134, -56.694032704868064, 64.88898127272921, -77.17295454529122, -77.17295454529122, 114.88896761960916, 64.88898127272921, -56.694032704868064, -2.682478888812399, 17.499980471114092, -56.694032704868064, -77.17295454529122, -117.37559051271262, -56.694032704868064, 9.677113743211134, -56.694032704868064, -56.694032704868064, 2.660273284119397, -2.682478888812399, -117.37559051271262, 64.88898127272921, -88.03328437216194, -22.675892549763255, 84.97879569969045, -49.69262779661864, 90.8774369507012, -21.423945954417384, 2.660273284119397, 9.677113743211134, 66.80834624286237, 2.660273284119397, -81.70500565381874, 64.88898127272921, -77.17295454529122, 9.677113743211134, 82.89645349128584, 64.88898127272921, 84.97879569969045, 47.39299013083392, -57.83798897319075, -69.56261557970399, 47.39299013083392, 75.80665842291098, 2.660273284119397, 64.88898127272921, 5.759918313758537, -32.8131109127397, -81.70500565381874, 66.80834624286237, 30.676357617444836, 9.677113743211134, 84.97879569969045, -77.17295454529122, 2.660273284119397, 17.499980471114092, 60.41884490473026, 84.97879569969045, -21.423945954417384, -37.952565639263085, 17.499980471114092, -57.83798897319075, 2.660273284119397, -56.694032704868064, 18.93629546008928, -21.503738588808243, 36.89181577133981, -44.74495929894548, 17.499980471114092, 17.499980471114092, 28.043293004197004, -77.17295454529122, 17.499980471114092, -37.952565639263085, 129.81231888533208, -9.699319347904137, -49.69262779661864, -56.694032704868064, 30.676357617444836, 9.677113743211134, -100.16877500488735, -77.17295454529122, -56.694032704868064, 9.677113743211134, 2.660273284119397, -117.37559051271262, 102.83242825363432, 84.97879569969045, 75.80665842291098, 17.499980471114092, 17.499980471114092, 90.8774369507012, 64.88898127272921, 64.88898127272921, -117.37559051271262, -92.55843603930012, -77.17295454529122, 2.660273284119397, -124.98592947829984, 18.93629546008928, 60.41884490473026, -57.83798897319075, -49.69262779661864, 41.04845252863289, 47.39299013083392, 17.499980471114092, 36.89181577133981, 2.660273284119397, -69.56261557970399, 90.8774369507012, 129.81231888533208, 28.043293004197004, -100.16877500488735, 5.759918313758537, 2.660273284119397, 64.88898127272921, -44.74495929894548, 17.499980471114092, 115.9916668031018, 9.677113743211134, 9.677113743211134, 129.81231888533208, -57.83798897319075, 114.88896761960916, 53.63196909653162, 64.88898127272921, -117.37559051271262, -56.694032704868064, 17.499980471114092, -77.17295454529122, -135.95151783810329, 17.499980471114092, 17.499980471114092, 85.53307475713086, 28.043293004197004, -56.694032704868064, 9.677113743211134, -14.071643847134714, 5.759918313758537, 90.8774369507012, 90.8774369507012, 72.37383620766339, 17.499980471114092, 17.499980471114092, 2.660273284119397, -56.694032704868064, 2.660273284119397, 2.660273284119397, -22.675892549763255, -37.952565639263085, -49.69262779661864, 17.499980471114092, -21.423945954417384, 5.759918313758537, -56.694032704868064, -22.675892549763255, 9.677113743211134, 64.88898127272921, 129.81231888533208, 17.499980471114092, 48.84680707427293, 82.89645349128584, -124.98592947829984, 60.41884490473026, -135.95151783810329, -77.17295454529122, 75.80665842291098, -14.071643847134714, -21.423945954417384, 66.80834624286237, 64.88898127272921, 47.39299013083392, -81.70500565381874, -37.952565639263085, -117.37559051271262, 60.41884490473026, -14.071643847134714, -81.70500565381874, 47.39299013083392, 64.88898127272921, 144.88309741312227, 17.499980471114092, -14.071643847134714, -56.694032704868064, -124.98592947829984, -88.03328437216194, 5.759918313758537, 17.499980471114092, 9.677113743211134, -56.694032704868064, 64.88898127272921, -57.83798897319075, 5.759918313758537, 41.04845252863289, 144.88309741312227, -26.003492233340502, -49.69262779661864, -37.952565639263085, 5.759918313758537, 9.677113743211134, 66.80834624286237, 2.660273284119397, -49.69262779661864, 60.41884490473026, -56.694032704868064, -77.17295454529122, -56.694032704868064, -56.694032704868064, 47.39299013083392, 17.499980471114092, 84.97879569969045, 17.499980471114092, -56.694032704868064, 17.499980471114092, -56.694032704868064, -22.675892549763255, -117.37559051271262, 36.89181577133981, -56.694032704868064, 9.677113743211134, 47.39299013083392, -22.675892549763255, 17.499980471114092, -26.003492233340502, 17.499980471114092, -21.423945954417384, -124.98592947829984, 59.34798143376704, 72.37383620766339, -68.42292059380576, -2.682478888812399, 9.677113743211134, 84.97879569969045, 53.63196909653162, 64.88898127272921, 60.41884490473026, 64.88898127272921, 17.499980471114092, -49.69262779661864, -21.423945954417384, 5.759918313758537, -68.42292059380576, 100.62057648997136, 47.39299013083392, 47.39299013083392, 47.39299013083392, 64.88898127272921, 17.499980471114092, 30.676357617444836, 9.677113743211134, 9.677113743211134, -56.694032704868064, -77.17295454529122, 59.34798143376704, -22.675892549763255, 60.41884490473026, -56.694032704868064, 28.043293004197004, -56.694032704868064, 64.88898127272921, 47.39299013083392, 2.660273284119397, 28.043293004197004, 2.660273284119397, 28.043293004197004, 2.660273284119397, 47.39299013083392, 5.759918313758537, -77.17295454529122, -56.694032704868064, 60.41884490473026, 2.660273284119397, -135.95151783810329, -21.423945954417384, -56.694032704868064, 5.759918313758537, 84.97879569969045, 17.499980471114092, -56.694032704868064, -77.17295454529122, 84.97879569969045, -117.37559051271262, 129.81231888533208, 5.759918313758537, 90.8774369507012, 64.88898127272921, 5.759918313758537, 60.41884490473026, 5.759918313758537, 17.499980471114092, 28.043293004197004, 72.37383620766339, -56.694032704868064, 60.41884490473026, 2.660273284119397, -92.55843603930012, -14.071643847134714, -37.952565639263085, 2.660273284119397, -77.17295454529122, -49.69262779661864, 114.88896761960916, -56.694032704868064, -124.98592947829984, 9.677113743211134, 17.499980471114092, 2.660273284119397, 18.93629546008928, 84.97879569969045, 17.499980471114092, 114.88896761960916, 47.39299013083392, -56.694032704868064, -56.694032704868064, -77.17295454529122, 85.53307475713086, 114.88896761960916] )
        # np.testing.assert_array_equal(canned_predictions,predictions)

    def test_performanceClassificationBagger(self):
        random.seed(101)

        # first create some data (classification)
        X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


        myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,
            X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

        classTrainingResults = myBagger.test(X_train,y_train)

        classTestResults = myBagger.test(X_test,y_test)
        featureImp = myBagger.calcFeatureImportance()

        # check feature importance agasint canned output
        canned_fimp = np.array([0.49633629, 0.50366371])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        #check training results 
        canned_training_meansure1 = 1.0
        canned_training_cmatrix = np.array([[226., 0., 0.] ,[ 0., 217., 0.,] ,[ 0., 0., 227.]])
        self.assertAlmostEqual(canned_training_meansure1,classTrainingResults.score[0])
        training_cmatrix = classTrainingResults.score[1]
        np.testing.assert_array_equal(canned_training_cmatrix,training_cmatrix.getConfusionMatrix())

        #check test results 
        canned_test_meansure1 = 0.996969696969697
        canned_test_cmatrix = np.array([[108., 0., 0.] ,[ 0. ,116., 0.] ,[ 1., 0., 105.]])
        self.assertAlmostEqual(canned_test_meansure1,classTestResults.score[0])
        test_cmatrix = classTestResults.score[1]
        np.testing.assert_array_equal(canned_test_cmatrix,test_cmatrix.getConfusionMatrix())

    def test_performanceRegressionBagger(self):
        random.seed(101)
        # first create some data (regression)
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
        
        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


        myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4,"include_OOB":True})

        # We should know the OOB before run any test : 
        # print(f"OOB Score is : {myBagger.OOBScore}") 
        
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
        print(f"feature imp: {featureImp}")

        # check feature importance agasint canned output
        canned_fimp = np.array([0.08254039 ,0.91745961])
        self.assertAlmostEqual(np.sum(canned_fimp-featureImp),0)

        # check training metrics
        canned_training_MSE = 228.55917134749154
        canned_training_R2 = 0.9584908233323536
        self.assertAlmostEqual(canned_training_MSE,regTrainingResults.score[1])
        self.assertAlmostEqual(canned_training_R2,regTrainingResults.score[0])

        # check test metrics
        canned_test_MSE = 407.530585018917
        canned_test_R2 = 0.9205705061600413
        self.assertAlmostEqual(canned_test_MSE,regTestResults.score[1])
        self.assertAlmostEqual(canned_test_R2,regTestResults.score[0])

    def test_proximity_regression(self):
        random.seed(101)

        # first create some data (regression)
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
        
        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

        myBagger = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4})

        proximityMatrix = myBagger.calcProximity(X_train)
        # just check agssint a couple of partial rows of the results 
        print(f"prox matrix row 0 looks like this {proximityMatrix[0,0:20]}")
        print(f"prox matrix row 1 looks like this {proximityMatrix[1,0:20]}")
        canned_proxMatrixTop1 = np.array([0., 0. ,0., 0., 0. ,0., 0. ,0. ,0.,0. ,0., 0. ,0. ,0.2 ,1. ,0. ,0. ,0. ,0. ,0. ])
        canned_proxMatrixTop2 = np.array([0., 0., 0.6, 0., 0., 0., 0., 0.2, 0., 0., 0., 0., 0.2, 0., 0., 0.8, 0., 0.0, 0.0, 0. ])
        np.testing.assert_array_equal(canned_proxMatrixTop1,proximityMatrix[0,0:20])
        np.testing.assert_array_equal(canned_proxMatrixTop2,proximityMatrix[1,0:20])

    def test_proximity_classificstion(self):
        random.seed(101)

        # first create some data (classification)
        X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

        # Split data into test and train - we'll take a third as test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 


        myBagger = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,5,X_train,y_train,**{"min_split":25,"max_depth":4})

        proximityMatrix = myBagger.calcProximity(X_train)
        # just check agssint a couple of partial rows of the results 
        print(f"prox matrix row 0 looks like this {proximityMatrix[0,0:20]}")
        print(f"prox matrix row 1 looks like this {proximityMatrix[1,0:20]}")
        canned_proxMatrixTop1 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1.] )
        canned_proxMatrixTop2 = np.array([0. ,0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0.8, 1., 1., 1., 1., 0., 0. ])
        np.testing.assert_array_equal(canned_proxMatrixTop1,proximityMatrix[0,0:20])
        np.testing.assert_array_equal(canned_proxMatrixTop2,proximityMatrix[1,0:20])


if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner())