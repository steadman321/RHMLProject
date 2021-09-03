##############################################################################################################
# Package Name  :     RegressionBooster.py
# Description   :     Provides boosting functionality for regression boosters
# ############################################################################################################
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
import numpy as np
from dataclasses import dataclass
from typing import List
from RHML.Ensembles.Boosters.BaseGradBooster import RHMLGradBooster

###############################################################################################
# Class Name    :       RHMLRegressionBooster
# Description   :       Boosting Ensemble (regression) Implementation
# I'm using B here as my MAX number of learners to add (notes say M - check the book!)
################################################################################################
class RHMLRegressionGradBooster(RHMLGradBooster):
    ###############################################################################################
    # Params        :
    #   B           : the number of base models to include
    #   samples     : the data to use to fit 
    #   labels      : the actual labels for the provided samples
    #   kwargs      : optional params and hyperparams (e.g include_oob). 
    #                 (Note these will get passed on to any base model created too)
    ################################################################################################
    def __init__(self,B,samples,labels,**kwargs):
        super(RHMLRegressionGradBooster,self).__init__(B,samples,labels,**kwargs)

    def _calcInitialPrediction(self,labels):
        return np.mean(labels)  

    #note: 'labels' are not original labels, but the new residuals  
    def _createBaseModel(self,samples,labels,**kwargs):
        return RHMLRegressionDecisionTree(samples,labels,**kwargs)
        
    ###############################################################################################
    # _calcTestScore : calc MSE and R2
    ############################################################################################### 
    def _calcTestScore(self,testPredictions,testActuals):
        diff = testActuals - testPredictions
        sqaureLoss= np.square(diff)
        totalSumSquares = np.sum(sqaureLoss)
        avgObservedValue = np.mean(testActuals)
        residuals = testActuals - avgObservedValue
        sqaureResiduals= np.square(residuals)
        sumOfResiduals = np.sum(sqaureResiduals)
        r_sqaured = 1 - (totalSumSquares/sumOfResiduals)
        # also calc MSE too :
        mse = np.mean(sqaureLoss)
        return (r_sqaured,mse)

    def getSummary(self):
        return []

    def getModelType(self):
        return "Gradient Booster"