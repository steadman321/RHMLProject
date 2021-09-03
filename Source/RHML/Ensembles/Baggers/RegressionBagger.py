##############################################################################################################
# Package Name  :     RegressionBagger.py
# Description   :     Extends the base RHMLBagger to provide regression specific implementation
# ############################################################################################################

from RHML.Ensembles.Baggers.BaseBagger import RHMLBagger
from scipy import stats
import numpy as np

###############################################################################################
# Class Name    :       RHMLRegressionBagger
# Description   :       Classification Bagging Ensemble Implementation 
################################################################################################
class RHMLRegressionBagger(RHMLBagger):

    def __init__(self,baseModel,B,samples,labels,**kwargs):
        super(RHMLRegressionBagger,self).__init__(baseModel,B,samples,labels,**kwargs)

    ###############################################################################################
    # _aggregatePredictions: for regression this is the mean of all the predictions produced 
    # by the individual models in the ensemble
    ###############################################################################################
    def _aggregatePredictions(self,predictions):
        m = np.mean(predictions)
        return m
        
    ###############################################################################################
    # _calcTestScore : calc MSE and R2
    ############################################################################################### 
    def _calcTestScore(self,testPredictions,testActuals):
        # For regression we use MSE : sum of diff sqaured
        # calc r^2  : 
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

    def getModelType(self):
        return "Regression Bagger"