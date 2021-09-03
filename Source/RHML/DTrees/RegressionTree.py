#####################################################################################################
# Package Name  :     RegressionTree.py
# Description   :     Extends the base RHMLDecisionTree to provide regression specific implementation
# ####################################################################################################

from RHML.DTrees.BaseTree import RHMLDecisionTree
import numpy as np

###############################################################################################
# Class Name    :       RHMLRegressionDecisionTree
# Description   :       Regression Descision Tree Implementation 
################################################################################################
class RHMLRegressionDecisionTree(RHMLDecisionTree):
    # Construtor : Need to provide set of data as a minimum to create a tree. The base tree owns the data. 
    def __init__(self,x_train,y_train,**kwargs):
        super(RHMLRegressionDecisionTree,self).__init__(x_train,y_train,**kwargs)

    ################################################################################################
    #  _calcPrediction:  for regression this is the mean of the observations in the node
    ###############################################################################################
    def _calcPrediction(self,predictionNode):
        theMean= np.mean(self._getNodeLabelData(predictionNode))
        return theMean

    ################################################################################################
    #  _calcImpurity : for regression this is using the RSS calculation
    ###############################################################################################
    def _calcImpurity(self,regionClasses):
        return self._calcRSS(regionClasses)

    ################################################################################################
    #  _calcRSS: used for impurity measure
    ###############################################################################################
    def _calcRSS(self,regionLables):
        if len(regionLables)==0:
            return 0
        # calc the mean of all labels : this is our prediction for this region 
        localPrediction = np.mean(regionLables)

        # Now need to look at all data points and do sqaure loss for them against above prediction : 
        adjustedLabels = regionLables-localPrediction
        sqaureLoss = np.square(adjustedLabels)
        sumSqauredLoss = np.sum(sqaureLoss)
        return sumSqauredLoss


    ################################################################################################
    #  _calcSplitCost : for regression this is the total impurity (RSS) across both regions
    ###############################################################################################
    def _calcSplitCost(self,splitSummary):
        return splitSummary.impurityLeft + splitSummary.impurityRight

    ################################################################################################
    #  _calcTestScore: 
    # for regression going to calc MSE and R^2
    ###############################################################################################
    def _calcTestScore(self,testPredictions,testActuals):
        # For regression we use MSE : sum of diff sqaured
        # # calc r^2  : 
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

    #############################
    # Public API 
    #############################
    def getModelType(self):
        return "Regression Decision Tree"

