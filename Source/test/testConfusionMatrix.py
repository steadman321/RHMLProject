
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


from RHML.DTrees.ConfusionMatrix import RHMLConfusionMatrix



def testConfusionMatrix():
    testActuals =     [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,0,1,1,2,2,2,0,0,0,1,1,2]
    testPredictions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2]

    cMatrix = RHMLConfusionMatrix(testPredictions,testActuals)

    microMetrics = cMatrix.calcMicroMetrics()
    macroMetrics = cMatrix.calcMacroMetrics()
    weightedMetrics = cMatrix.calcWeightedMetrics()

    expectedMicroMetrics    = (0.2777777777777778, 0.2777777777777778, 0.2777777777777778)
    expectedMacroMetrics    = (0.2638888888888889, 0.2933177933177933, 0.242495126705653)
    expectedWeightedMetrics = (0.2604166666666667, 0.2777777777777778, 0.23430799220272902)

    print(f"{cMatrix.getConfusionMatrix()}")
    print(f"{cMatrix.getMetrics()}")
    print(f"micro : {microMetrics}")
    print(f"macro : {macroMetrics}")
    print(f"weighted : {weightedMetrics}")
    print(f"summary : {cMatrix.getMetricsSummary()}")


    if (expectedMicroMetrics == microMetrics) and (expectedMacroMetrics == macroMetrics) and (expectedWeightedMetrics==weightedMetrics):
        passed=True
    else:
        passed=False

    print(f"Test Passed: {passed}")


testConfusionMatrix()