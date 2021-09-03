##############################################################################################################
# Package Name  :     BaseGradBooster.py
# Description   :     Provides core gradient boosting functionality for thre RHML booster classes
# ############################################################################################################
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
import numpy as np
from dataclasses import dataclass
from typing import List

###############################################################################################
# Class Name    :       RHMLGradBooster
# Description   :       Boosting Ensemble (base) Implementation 
# I'm using B here as my MAX number of learners to add i.e number of base models
################################################################################################
class RHMLGradBooster():
    ###############################################################################################
    # Params        :
    #   B           : the number of base models to include
    #   samples     : the data to use to fit 
    #   labels      : the actual labels for the provided samples
    #   kwargs      : optional params and hyperparams (e.g include_oob). 
    #                 (Note these will get passed on to any base model created too)
    ################################################################################################
    def __init__(self,B,samples,labels,**kwargs):
        self.B = B
        self.samples = samples
        self.labels = labels

        self.learningRate = 0.1

        # initialise an empty set of base models
        self.ensembleModels = []

        # step 1: calc the initial prediction based on the provided labels : this is diff per classification and regressoin
        self.initalPrediction = self._calcInitialPrediction(self.labels)

        # we use the same prediction for all samples to begin with : 
        self.predictions = np.full(self.labels.shape,self.initalPrediction)

        # step 2: loop to improve predictions:
        for b in range(self.B):
            # step 2a : compute the residuals (comparing observed with latest predictions)
            self.residuals = self._calcResiduals()

            # step 2b : Fit new base model to predict modified residuals : NOTE: not using original labels here - residuals are now in place of labels
            newBaseModel = self._createBaseModel(self.samples,self.residuals,**kwargs)

            # step 2c : Calc new predictions (diff for regression and classification : reg = avg in region!)
            # This is not actually required as a sep step, since the base model will take care of this!
            # i.e as part of building the new base model, each region will have its own prediction 

            # step 2d : Update the model with a learning-rate X new model , and make new predictions for all samples
            #  2 substeps here : i)add the new model to the list  ii)calc new predicitons for all samples, using ALL the trees!
            self.ensembleModels.append(newBaseModel)

            # now need to get set of predictions for our samples with the new tree, and then add a part of them to our running predictions .... 
            newRawPredictions = np.array(newBaseModel.predict(self.samples))
            newRawPredictions = newRawPredictions * self.learningRate

            self.predictions = self.predictions+newRawPredictions

        # Model is cooked and ready to be used for predictions .... 

    ################################################################################################
    # _calcResiduals : we use (Observed - Predicted) to calc this
    ################################################################################################
    def _calcResiduals(self):
        return self.labels-self.predictions


    ################################################################################################
    # public api
    ################################################################################################

    ###############################################################################################
    # predict : calculate a prediction for a given set of test observations  
    # push each observation down each tree, and aggregate the results of all trees
    ###############################################################################################
    def predict(self,testObservations): 
        results = []
        # for observation in testObservations:
        predictionsTotal = np.full(testObservations.shape[0],self.initalPrediction)

        # for each tree in the ensemble - get predictions for all the observations at once : 
        for b in self.ensembleModels:
            bRawPredictions = np.array(b.predict(testObservations))
            
            # apply learning rate for each models prediction
            bRawPredictions = bRawPredictions * self.learningRate 

            # add this to a running total of predictions per observation 
            predictionsTotal=predictionsTotal+bRawPredictions

        return predictionsTotal

    ################################################################################################
    # test : 
    # using a fully fitted booster, calculate the predictions for a given set of observations, 
    # and compare to the provided actual labels. Return a test score.
    # 
    # testObservations : an array like structure , columns are features, rows are observations
    # testLabels : an array: one row per observation : actual label 
    #################################################################################################
    def test(self,testObservations,testLabels):
        # First we get predictions for our test observations
        testPredictions = self.predict(testObservations)

        # now we need to compate our original testObservations and our predictions 
        score = self._calcTestScore(testPredictions,testLabels)

        result = testResults(predictions=testPredictions,score=score)
        return result

    ################################################################################################
    # calcFeatureImportance : 
    # Note: when calling calcFeatureImportance on the underlying model, we need to turn off normalisation
    # as that is done at this level across all models in the ensemble. 
    ################################################################################################
    def calcFeatureImportance(self):
        # need to calc feature importance for each base learner , then get the mean of those
        featureImportance = np.mean([bag.calcFeatureImportance(False) for bag in self.ensembleModels],axis=0)
        # now normalise after avg ; 
        totalGain = np.sum(featureImportance)
        featureImportance = featureImportance/totalGain

        print(f"GradBooster: Feature importance : {featureImportance} ")
        return featureImportance

@dataclass
class testResults:
    """Results from running test(), returns predictions and overall score"""
    predictions: List[float]
    score: float