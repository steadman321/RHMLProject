##############################################################################################################
# Package Name  :     BaseBagger.py
# Description   :     Provides core bagging functionality for thre RHML bagging classes
# 
# Note: This is an abstract class and needs to be extended to provide certain metods 
# such as _aggregatePredictions and _calcTestScore
# ############################################################################################################
from RHML_CMD.rhml_configreader import findDataToLoad
import numpy as np
from dataclasses import dataclass
from typing import List

from RHML.Utils.Bootstrap import RHMLBootstrap,RHMLBootstrap_withOOB
import RHML.Utils.Proximity as RHMLProximity

###############################################################################################
# Class Name    :       RHMLBagger
# Description   :       Bagging Ensemble Implementation 
# Params        :
#                   baseModel   : the type of model to use as the base model 
#                   B           : the number of base models to include
#                   samples     : the data to use to fit 
#                   labels      : the actual labels for the provided samples
#                   kwargs      : optional params and hyperparams (e.g include_oob). 
#                                 (Note these will get passed on to any base model created too)
################################################################################################
class RHMLBagger():

    def __init__(self,baseModel,B,samples,labels,**kwargs):
        # cached model and data
        self.B = B
        self.samples = samples
        self.labels = labels
        self.baseModel = baseModel

        # OOB cache and flags
        self.include_OOB = kwargs.get('include_oob',False)
        self._OOB_contribution_index = []
        self._OOB_contribution_prediction = []
        self.OOBScore = None

        # initialise an empty set of base models
        self.ensembleModels = []

        for b in range(B):
            # first we need to bootstrap our data
            if self.include_OOB:
                bootstrapSamples,boostrapLabels, OOBIndexes = RHMLBootstrap_withOOB(samples,labels)
            else:
                bootstrapSamples,boostrapLabels = RHMLBootstrap(samples,labels)

            # now create another base model  
            newBaseModel= self._createBaseModel(bootstrapSamples,boostrapLabels,**kwargs)
            self.ensembleModels.append(newBaseModel)

            # if we are looking to do OOB, then calc the OOB predictions for this latest base model
            if self.include_OOB:
                self._calcOOBContributions(newBaseModel, sorted(OOBIndexes))

        if self.include_OOB:
            self._calcOOBError()

    # This can be overwritten by child classes if other logic needed here e.g randomise features to use, for example!
    def _createBaseModel(self,bootstrapSamples,boostrapLabels,**kwargs):
        return self.baseModel(bootstrapSamples,boostrapLabels,**kwargs)

    ###############################################################################################
    # _aggregatePredictions : 
    # Abstract : this method needs to be provided by exteded classes
    # This method will implement the logic for deciding what the final ensemble prediction is 
    # given all the contributing base model predictions
    ###############################################################################################
    def _aggregatePredictions(self,predictions):
        raise NotImplementedError("RHMLBagger: _aggregatePredictions: Subclasses should implement this!")

    ###############################################################################################
    # _calcTestScore : 
    # Abstract : this method needs to be provided by exteded classes
    # This method will implement the logic for calculating overall test score based on predictions 
    # and actual labels
    ###############################################################################################
    def _calcTestScore(testPredictions,testLabels):
        raise NotImplementedError("RHMLBagger: _calcTestScore: Subclasses should implement this!")
    
    ###############################################################################################
    # _calcOOBContributions:
    # This calcs the OOB predictions for a given base model and adds to the OOB cache
    ###############################################################################################
    def _calcOOBContributions(self,newBaseModel, OOBIndexes):
        OOBSamples = self.samples[OOBIndexes]
        OOBPredictions = newBaseModel.predict(OOBSamples)
        self._OOB_contribution_index.extend(OOBIndexes)
        self._OOB_contribution_prediction.extend(OOBPredictions)

    ###############################################################################################
    # _calcOOBError : 
    # After all OOB contributions are known, use underling base models to make predictions and calc overall score for OOB
    ###############################################################################################
    def _calcOOBError(self):
        OOB_final_predictions = []
        OOB_actual_labels = []
        uniqueOOBIndexes = np.unique(self._OOB_contribution_index)
        for index in uniqueOOBIndexes:
            # get the predictions we are after for this OOB sample
            subIndexes = self._OOB_contribution_index==index
            subsetPredictions = np.array(self._OOB_contribution_prediction)[subIndexes]
            # calc the aggregated prediction for this OOB sample 
            aggPrediction = self._aggregatePredictions(subsetPredictions)
            # cache this 
            OOB_final_predictions.append(aggPrediction)
            OOB_actual_labels.append(self.labels[index])

        # calc overall OOB score using underling _calcTestScore methods
        self.OOBScore = self._calcTestScore(np.array(OOB_final_predictions),np.array(OOB_actual_labels))
    ###############################################################################################
    # public api
    ###############################################################################################

    ###############################################################################################
    # predict : calculate a prediction for a given set of test observations  
    # push each observation down each tree, and aggregate the results of all trees
    # this depeends on extended classes implementtion of _aggregatePredictions which will 
    # be called for each model in the ensemble
    ###############################################################################################
    def predict(self,testObservations): 
        results = []

        # push each observaton down each tree, and aggregate the results of all trees
        for observation in testObservations:
            predictions = []
            for bag in self.ensembleModels:
                prediction = bag.predict([observation])[0]
                predictions.append(prediction)

            # aggregsation depends on type of problem : so not done in base class!
            results.append(self._aggregatePredictions(predictions))
        return results
    
    ################################################################################################
    # test : 
    # using a fully fitted bagger, calculate the predictions for a given set of observations, 
    # and compare to the provided actual labels. Return a test score.
    # 
    # testObservations : an array like structure , columns are features, rows are observations
    # testLabels : an array: one row per observation : actual label 
    #################################################################################################
    def test(self,testObservations,testLabels):
        # First we get predictions for our test observations
        testPredictions = self.predict(testObservations)

        # now we need to compate our original testObservations and our predictions 
        # NOTE: the _calcTestScore will not be defined in base class
        score = self._calcTestScore(testPredictions,testLabels)

        result = baggerTestResults(predictions=testPredictions,score=score)
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

        return featureImportance

    ################################################################################################
    # calcProximity : 
    # For a fully fitted bagger, use the RHMLProximity class to calcualte a proximity matrix.  
    ################################################################################################
    def calcProximity(self,observations):
        return RHMLProximity.calcProximity(self.ensembleModels,observations)

    def getSummary(self):
        return []

    def getModelType(self):
        return "Base Bagger"


@dataclass
class baggerTestResults:
    """Results from running test(), returns predictions and overall score"""
    predictions: List[float]
    score: float





