##############################################################################################################
# Package Name  :     ClassificationTree.py
# Description   :     Extends the base RHMLDecisionTree to provide classification specific implementation
# ############################################################################################################
from RHML.DTrees.BaseTree import RHMLDecisionTree, splitResults, splitSummary
from RHML.DTrees.ConfusionMatrix import RHMLConfusionMatrix
import numpy as np
from scipy import stats

###############################################################################################
# Class Name    :       RHMLClassificationDecisionTree
# Description   :       Classification Descision Tree Implementation 
################################################################################################
class RHMLClassificationDecisionTree(RHMLDecisionTree):
    # Construtor : Need to provide set of data as a minimum to create a tree. The base tree owns the data. 
    def __init__(self,x_train,y_train,**kwargs):
        super(RHMLClassificationDecisionTree,self).__init__(x_train,y_train,**kwargs)

    ################################################################################################
    # _calcPrediction : mode of labels in a region
    ################################################################################################
    def _calcPrediction(self,predictionNode):
        # For classification , the prediction is the most frequent of the observations in the node, i.e the mode
        return stats.mode(self._getNodeLabelData(predictionNode),axis=None)[0][0]

    ################################################################################################
    # _calcImpurity: based on gini index
    ################################################################################################
    def _calcImpurity(self,regionClasses):
        # For classification this is based on the Gini index
        return self._calcGini(regionClasses)

    ################################################################################################
    # _calcSplitCost : calculate the total cost for a given split point
    ################################################################################################
    def _calcSplitCost(self,splitSummary):
        N = splitSummary.leftRegionCount+splitSummary.rightRegionCount
        return (
                    ((splitSummary.leftRegionCount/N) * splitSummary.impurityLeft ) + 
                    ((splitSummary.rightRegionCount/N)*splitSummary.impurityRight )
                )

    ################################################################################################
    # _calcTestScore : get metrics for test score
    # For classification going to create a multiclass confusion matrix , 
    # then use that to show : accuracy, precision, recall, F1
    ################################################################################################
    def _calcTestScore(self,testPredictions,testActuals):
        # simple accuracy measure 
        predictedCorrectly = [testPredictions == testActuals]
        accuracy = (np.sum(predictedCorrectly))/len(testActuals)

        # This confusionMatrix will also have a bunch of related metrics inside it
        cMatrix = RHMLConfusionMatrix(testPredictions,testActuals)

        return (accuracy,cMatrix)
    
    ################################################################################################
    # _calcGini : calculate the total cost for a given split point
    ###############################################################################################
    def _calcGini(self,regionClasses):
        # for each class, need to know its frequency 
        (unique, counts) = np.unique(regionClasses, return_counts=True)
        sumCounts = np.sum(counts)
        countsProportions = counts/sumCounts
        gini = np.sum(countsProportions*(1-countsProportions))

        return gini
    
    ################################################################################################
    # _calcLabelProportions: calc proportion per unique class in a region
    ################################################################################################
    def _calcLabelProportions(self,regionClasses):
        (unique, counts) = np.unique(regionClasses, return_counts=True)
        sumCounts = np.sum(counts)
        countsProportions = counts/sumCounts
        return unique,countsProportions

    #############################
    # Public API 
    #############################
    # returns unique, counts : i.e the labels are listed, then the props
    def calcClassProbabilities(self,observation):
        predictionNode = self._findPredictionNode(observation)
        regionLabels = self._getNodeLabelData(predictionNode)
        return self._calcLabelProportions(regionLabels)

    def getModelType(self):
        return "Classification Decision Tree"

        