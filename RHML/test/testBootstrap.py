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
import random
import numpy as np

def simpleTest():
    # create a simple test data set (a numpy array)

    # set a seed if want same numbers each time .... 
    random.seed(101)
    testData = np.arange(10)
    print(f"Test data looks like this       {testData}")
    bootstrap = RHMLBootstrap(testData)
    print(f"Boostrap data looks like this   {bootstrap}")

def dataLabelSplitTest():
    # create a simple test data set (a numpy array)

    # set a seed if want same numbers each time .... 
    random.seed(101)
    testData = np.arange(10)
    testLabels = np.arange(10)
    print(f"Test data looks like this         {testData}")
    print(f"Test lables looks like this       {testLabels}")

    bootstrapData,boostrapLabels = RHMLBootstrap(testData,testLabels)
    print(f"Boostrap data looks like this   {bootstrapData}")
    print(f"Boostrap lables looks like this   {boostrapLabels}")

def simpleWeightedTest():
    # set a seed if want same numbers each time .... 
    random.seed(101)
    testData = np.linspace((1,2),(10,20),10) #np.arange(10)
    testLabels = np.arange(10)
    testLabels= testLabels.reshape((10, 1))
    testWeights = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    testWeights= testWeights.reshape((10, 1))
    print(f"Test data looks like this         {testData}")
    print(f"Test lables looks like this       {testLabels}")

    bootstrapData,boostrapLabels = RHMLWeightedBoostrap(testData,testLabels,testWeights
    )
    print(f"Boostrap data looks like this   {bootstrapData}")
    print(f"Boostrap labels looks like this   {boostrapLabels}")

# RHMLWeightedBoostrap

# simpleTest()
# dataLabelSplitTest()
simpleWeightedTest()

