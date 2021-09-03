import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
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

from RHML.Utils.Bootstrap import RHMLBootstrap,RHMLWeightedBoostrap

class TestBoostrapper(unittest.TestCase):
    def test_simpleBoostrap(self):
        # set a seed if want same numbers each time .... 
        random.seed(101)
        testData = np.arange(10)
        testLabels = np.arange(10)

        bootstrapData,boostrapLabels = RHMLBootstrap(testData,testLabels)

        # verify output
        canned_test_data=np.array([0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9])
        canned_test_labels=np.array([0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9])
        canned_boot_data=np.array([9 ,3, 8, 5, 7, 0, 8, 3, 9, 3])
        canned_boot_labels=np.array([9, 3, 8, 5, 7, 0, 8, 3, 9, 3])

        np.testing.assert_array_equal(canned_test_data,testData)
        np.testing.assert_array_equal(canned_test_labels,testLabels)
        np.testing.assert_array_equal(canned_boot_data,bootstrapData)
        np.testing.assert_array_equal(canned_boot_labels,boostrapLabels)

    def test_weightedBoostrap(self):
        # set a seed if want same numbers each time .... 
        random.seed(101)
        testData = np.linspace((1,2),(10,20),10) #np.arange(10)
        testLabels = np.arange(10)
        testLabels= testLabels.reshape((10, 1))
        testWeights = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        testWeights= testWeights.reshape((10, 1))

        bootstrapData,boostrapLabels = RHMLWeightedBoostrap(testData,testLabels,testWeights)

        # verify output
        canned_boot_data=np.array([[ 6. ,12.] ,[ 2., 4.] ,[10., 20.] ,[10., 20.] ,[ 5. ,10.] ,[ 7., 14.], [ 3., 6.] ,[ 3., 6.] ,[ 3. ,6.] ,[ 7., 14.]] )
        canned_boot_labels=np.array([5., 1., 9., 9. ,4. ,6. ,2., 2. ,2. ,6.])
        np.testing.assert_array_equal(canned_boot_data,bootstrapData)
        np.testing.assert_array_equal(canned_boot_labels,boostrapLabels)





if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner())