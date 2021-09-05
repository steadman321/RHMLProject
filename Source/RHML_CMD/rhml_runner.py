################################################################################################
# rhml_runner
# 
# This will coordinate a set of model creation and test runs based on passed in config
###############################################################################################
from os import system
import sys
import configparser
import math
import random
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import rhml_dataloader,rhml_reporter,rhml_configreader
from RHML.DTrees import ClassificationTree as CTree 
from RHML.DTrees import RegressionTree as RTree
from RHML.Ensembles.RandomForest import ClassificationRandomForest
from RHML.Ensembles.RandomForest import RegressionRandomForest

from RHML.Ensembles.Baggers import ClassificationBagger,RegressionBagger
from RHML.DTrees.ClassificationTree import RHMLClassificationDecisionTree
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
from RHML.Ensembles.Boosters import AdaBooster
from RHML.Ensembles.Boosters import RegressionGradBooster

# Some public statics for runtypes supported
RUN_SINGLE  = 0
RUN_GRID    = 1
RUN_MULTI   = 2

#****************************************************************************************
# run
# Main entry point . Load required data and initiate the appropriate run type
# ***************************************************************************************
def run(configReader,runType):
    # Load the data we want to use : 
    dataName = rhml_configreader.findDataToLoad(configReader)[0][0]
    featuresToSkip = rhml_configreader.findFeaturesToSkip(configReader)
    theData,theLabels,theFeatures = rhml_dataloader.loadDataSet(dataName,featuresToSkip)

    # What type of problem are we dealing with : classification or regression -  This is driven the dataset selected
    problemType = rhml_dataloader.getDataConfig(dataName,rhml_dataloader.DATA_PROBLEM_TYPE)

    # Set random see so get similar results to compare - can be set in config
    random.seed(getRandomStateSeed(configReader))

    if runType==RUN_GRID:
        runGridSearch(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader)
    elif runType==RUN_MULTI:
        runMultiModel(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader)
    else:
        runSingleSplit(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader)

#****************************************************************************************
# runSingleSplit 
# Supports : one data source, multiple models, (no moving paramters)
# Creates : one report per model
# ***************************************************************************************
def runSingleSplit(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader):
    # data split for training and test
    X_train, X_test, y_train, y_test = train_test_split(theData, theLabels, test_size=0.33, random_state=getRandomStateSeed(configReader),shuffle=True)

    # get list of models we want to run
    modelsToRun = rhml_configreader.findWhichModelsToRun(configReader)

    reportTitleRoot= "Title : "

    # get title from config if we have one set
    reportTitleRoot = rhml_configreader.getTitleFromConfig(configReader)
    reportDesc = rhml_configreader.getDescriptionFromConfig(configReader)
    


    for model in modelsToRun:

        # get model specific params here : combine with defaults too 
        modelParamsString = model[0].upper()+"_PARAMS"
        modelKwargs = rhml_configreader.getModelKwargs(configReader,modelParamsString)


        # create the appropraite model with provided and default params
        theModel = createModel(problemType,model[0],X_train,y_train,modelKwargs)

        # create a report for this model - this gets training and test based results
        # note - this will open it in a browser asa get the report
        if theModel is not None:
            
            modelSummary,featureImp,training_results,testing_results,modelOOBResults = runTests(theModel,X_train, X_test, y_train, y_test)  

            # Some things report more things .... eg. proximity matrix for RF and Bags ... 
            proximityMatrix= calcProximityMatrix(theModel,X_train)

            reportTitle = reportTitleRoot+" - "+theModel.getModelType()
            rhml_reporter.buildReport(  dataName,
                                        problemType,
                                        reportTitle,
                                        reportDesc,
                                        theFeatures,
                                        featuresToSkip,
                                        modelKwargs,
                                        modelSummary,
                                        featureImp,
                                        training_results,
                                        testing_results,
                                        proximityMatrix,
                                        y_train,
                                        modelOOBResults)


#****************************************************************************************
# runGridSearch 
# Supports : one model type, but one or more hyperparams
#   Uses cross-validation to find the best setting for those hyperparams. 
#   The config needs to give a set of values  e.g[1,2,3] for a param that will get tested here
# Creates : a single report
# ***************************************************************************************
def runGridSearch(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader):
    
    # For grid search we need to get some specific config things from config file

    # Find the model to use (inside gridsearch config section)and check we only have one model to run for grid search!
    modelsToRun = rhml_configreader.findWhichGridModelsToRun(configReader)

    if len(modelsToRun)==0:
        sys.exit(f"There are no models listed in the grid search config section")
    
    if len(modelsToRun)>1:
        sys.exit(f"There are too many models listed in the grid search config section {modelsToRun} : Grid Search only supports one model at a time.")

    modelCode = (modelsToRun[0][0]).lower()
    print(f"Starting Grid Search for model {modelCode}")

    # get grid specific params: merge grid and standadrd settings for given model
    modelParamsString = modelCode.upper()+"_PARAMS"
    modelKwargs = rhml_configreader.getModelKwargs(configReader,modelParamsString)
    modelKwargs =rhml_configreader.adjustKwargsForGridSearch(configReader,modelKwargs)

    # Since we use these for reporting ensure kwargs carry defaults at this point too
    amendForDefaults(problemType,modelCode,modelKwargs)
    print(f"kwargs set to: {modelKwargs}")

    # Process the settings that have some 'ranges' 
    searchParamGrid = {}
    for key,value in modelKwargs.items():
        if isinstance(value,List):
            print(f"We have a range for : {key} and it looks like this  {value}")
            searchParamGrid[key]=value

    numberOfGridSearchParams = len(searchParamGrid)
    if numberOfGridSearchParams==0:
        sys.exit("For a grid search there needs to be at least one ranged hyperparamter")
    
    
    # Now calc all the pos combos of params we need to test in the grid search : 
    grid = ParameterGrid(searchParamGrid)

    # Split out data - use training data for CV, and eventually test on test only!
    X_train, X_test, y_train, y_test = train_test_split(theData, theLabels, test_size=0.33, random_state=getRandomStateSeed(configReader),shuffle=True)

    grid_results = []
    # We spin for as many combos in our grid .... 
    for index, gridItem in enumerate(grid):
        # The gridItem gives us the settings we need to change to our modelKwargs for this run
        runParams = modelKwargs.copy()   # start with x's keys and values
        runParams.update(gridItem)
        print(f"index:{index}: This run will use the following kwargs {runParams}")

        # start each step in the grid at the same place so compare like with like
        random.seed(getRandomStateSeed(configReader))
    
        cross_validation_folds = 5
        # split the data into cross_validation_folds : this just creates the index of the data in each fold!
        foldDataIndexes = createCVFoldDataIndexes(X_train.shape[0],cross_validation_folds)

        crossValScores = []
        for iteration in range(cross_validation_folds):
            trainingIndexs, testIndexes = getTrainTestIndexesForCVIteration(iteration, foldDataIndexes)
            cv_trainingData = X_train[trainingIndexs,:]
            cv_testData = X_train[testIndexes,:]
            cv_trainingLabels = y_train[trainingIndexs]
            cv_testLabels = y_train[testIndexes]        
        
            # create the appropraite model with provided and default params
            theModel = createModel(problemType,modelCode,cv_trainingData,cv_trainingLabels,runParams)
            if theModel is not None:
                test_results = theModel.test(cv_testData,cv_testLabels)

                # stash the score (which score to use?) in a 'results' table, with the key is the index of the ParamterGrid being used here!
                score = test_results.score[0]
                crossValScores.append(score)

        # now add the average across all cross validation runs as the final score for this combo of params
        grid_results.append(np.mean(crossValScores))


    print(f"The finals scores are  {grid_results}, sizee: {len(grid_results)} max : {max(grid_results)}")
    # want to know the index of the max, so we can get the settings that are best
    best_index = grid_results.index(max(grid_results))
    print(f"The best is at index = {best_index}")
    print(f"Settings to use are : {grid[best_index]}")

    # Report and plots ..... 
    reportTitleRoot = "Grid Search Report : "
    # get title from config if we have one set
    reportDesc = rhml_configreader.getDescriptionFromConfig(configReader)
    configTitle = rhml_configreader.getTitleFromConfig(configReader)
    if configTitle !="":
        reportTitleRoot = configTitle+" - "+reportTitleRoot

    reportTitle = reportTitleRoot+theModel.getModelType()
    rhml_reporter.buildGridSearchReport( dataName,featuresToSkip,reportTitle,reportDesc,searchParamGrid,grid,grid_results,best_index,modelKwargs)

#****************************************************************************************
# runMultiModel : Targetting commpare multiple models
# Supports : one dataset, multiple models with a single moving parameter
#   Only one moving paramter allowed at once 
#   One or more models allowed
#   Train a model based on base params and the moving parameter, and output results based on train/test data 
# Creates : a single report
# ***************************************************************************************
def runMultiModel(dataName,featuresToSkip,theData,theLabels,theFeatures,problemType,configReader):
    # For multi model we need to get some specific config things from config file

    # Find the models to use (inside multi model config section)
    modelsToRun = rhml_configreader.findWhichMultiModelsToRun(configReader)
    numberOfModels = len(modelsToRun)

    if numberOfModels==0:
        sys.exit(f"There are no models listed in the grid search config section")

    modelKwargs = rhml_configreader.getMultiModelKwargs(configReader)
    # Need pay special attention to which config params are 'moving' i.e have a list of values ... 
    movingParameters,movingParamaterValues = getListParams(modelKwargs)

    if movingParamaterValues is None or len(movingParamaterValues)==0:
        sys.exit(f"There are no ranged paramters in the multi-model section")

    numberParameterValues = len(movingParamaterValues[0])

    numberLists = len(movingParameters)
    if numberLists==0:
        sys.exit("For a multi model report there needs to be at a ranged hyperparamter e.g config with [1,2,3] format")
    elif numberLists > 1:
        sys.exit("For a multi model report there can only be one hyperparamter that has a range/list value e.g config with [1,2,3] format")

    # split the data in train / test
    X_train, X_test, y_train, y_test = train_test_split(theData, theLabels, test_size=0.33, random_state=getRandomStateSeed(configReader),shuffle=True)

    # set up some empty cache to  store the results 
    training_results_cache = np.zeros((numberOfModels,numberParameterValues))
    testing_results_cache = np.zeros((numberOfModels,numberParameterValues))
    oob_results_cache = np.zeros((numberOfModels,numberParameterValues))
    models_run=[]
    model_kwargs=[] #cache the actual kwargs used so we can report on them later

    # loop for each model
    for i,model in enumerate(modelsToRun):

        # start each step in the grid at the same place so compare like with like
        random.seed(getRandomStateSeed(configReader))

        # get raw params together for the model type
        modelCode = (model[0]).lower()
        models_run.append(modelCode)
        print(f"Starting Multi Model for model {modelCode} - model index is {i}")

        # get grid specific params: merge multi and standadrd settings for given model
        modelParamsString = modelCode.upper()+"_PARAMS"
        modelKwargs = rhml_configreader.getModelKwargs(configReader,modelParamsString)

        # renove the param we will override, as just want to record the 'other' params for reports
        adjusted_kwargs = dict(modelKwargs) #make deep copy
        adjusted_kwargs.pop(movingParameters[0],None)
        model_kwargs.append(adjusted_kwargs)

        print(f"kwargs is set to: {modelKwargs}")

        # now loop over the moving paramter values, and replace where needed 
        for p,param_value in enumerate(movingParamaterValues[0]):
            # Update kwargs to use the current value of the moving param
            modelKwargs[movingParameters[0]]=param_value

            theModel = createModel(problemType,modelCode,X_train,y_train,modelKwargs)
            if theModel is not None:
                train_results = theModel.test(X_train,y_train)
                test_results = theModel.test(X_test,y_test)

                # stash the score (which score to use?) in a 'results' table, with the key is the index of the ParamterGrid being used here!
                train_score = train_results.score[0]
                test_score = test_results.score[0]
                training_results_cache[i,p]=train_score
                testing_results_cache[i,p]=test_score

                # may have OOB results for this model too : if supported and if switched on 
                modelOOBResults = getattr(theModel,"OOBScore",None)
                if modelOOBResults is not None:
                    oob_results_cache[i,p]=modelOOBResults[0]
                else:
                    oob_results_cache[i,p]=-1

    print(f"Results are in ..... ")
    print(f"Training results : {training_results_cache}")
    print(f"Testing results : {testing_results_cache}")

    # now plot it : note this would be in a report at some point instead but just plot it for now
    # plotMultiModel(models_run,movingParameters[0],movingParamaterValues[0],training_results_cache,testing_results_cache)

    reportTitle = "Multi Model Report - "
    reportDesc = rhml_configreader.getDescriptionFromConfig(configReader)
    configTitle = rhml_configreader.getTitleFromConfig(configReader)
    if configTitle !="":
        reportTitle = reportTitle+" "+configTitle
    rhml_reporter.buildMultiModelReport( dataName,featuresToSkip,reportTitle,reportDesc,models_run,model_kwargs,movingParameters[0],movingParamaterValues[0],training_results_cache,testing_results_cache,oob_results_cache)
#****************************************************************************************#****************************************************************************************


#****************************************************************************************
# Supporting functions ..... 
#****************************************************************************************
# Depending on config, create the required model to run the job
def createModel(problemType, modelCode, data, labels,modelKwargs):
    if problemType == rhml_dataloader.CLASSIFICATION:
        return createClassificationModel(modelCode, data, labels,modelKwargs)
    else:
        return createRegressionModel(modelCode, data, labels,modelKwargs)

def createClassificationModel(modelCode, data, labels,modelKwargs):
    
    # All models except DT need 'number of trees' set , so check here
    numberTrees=modelKwargs.get(rhml_configreader.MODEL_NUM_TREES,0)
    if numberTrees == 0 and modelCode != rhml_configreader.MODEL_DTREE:
        print(f"MISSING CONFIG : Required config missing for model {modelCode} : {rhml_configreader.MODEL_NUM_TREES} must be > 0")
        return None

    if modelCode == rhml_configreader.MODEL_DTREE:
        model = CTree.RHMLClassificationDecisionTree(data,labels,**modelKwargs)
        return model            

    elif modelCode == rhml_configreader.MODEL_BAG:
        model = ClassificationBagger.RHMLClassificationBagger(RHMLClassificationDecisionTree,numberTrees,data,labels,**modelKwargs)
        return model

    elif modelCode == rhml_configreader.MODEL_RF:
        # check the must have params : could do this as validation step before run at all?
        numberFeatures=modelKwargs.get(rhml_configreader.MODEL_RF_NUM_FEATURES,0)
        if numberFeatures == 0:
            print(f"Required config missing for model {modelCode} : {rhml_configreader.MODEL_RF_NUM_FEATURES} must be > 0")
            return None
        myRandomForest = ClassificationRandomForest.RHMLClassificationRandomForest(numberFeatures,numberTrees,data,labels,**modelKwargs)
        return myRandomForest

    elif modelCode == rhml_configreader.MODEL_BOOST:
        if not modelKwargs.get('max_depth'):
            modelKwargs['max_depth'] = AdaBooster.RHMLAdaBooster.DEFAULT_MAX_DEPTH
        model = AdaBooster.RHMLAdaBooster(numberTrees,data,labels,**modelKwargs)
        return model


    return None

def createRegressionModel(modelCode, data, labels,modelKwargs):

    # All models except DT need 'number of trees' set , so check here
    numberTrees=modelKwargs.get(rhml_configreader.MODEL_NUM_TREES,0)
    if numberTrees == 0 and modelCode != rhml_configreader.MODEL_DTREE:
        print(f"MISSING CONFIG : Required config missing for model {modelCode} : {rhml_configreader.MODEL_NUM_TREES} must be > 0")
        return None
    
    if modelCode == rhml_configreader.MODEL_DTREE:
        model = RTree.RHMLRegressionDecisionTree(data,labels,**modelKwargs)
        return model            

    elif modelCode == rhml_configreader.MODEL_BAG:
        model = RegressionBagger.RHMLRegressionBagger(RHMLRegressionDecisionTree,numberTrees,data,labels,**modelKwargs)
        return model

    elif modelCode == rhml_configreader.MODEL_RF:
        # check the must have params
        numberFeatures=modelKwargs.get(rhml_configreader.MODEL_RF_NUM_FEATURES,0)
        if numberFeatures == 0:
            print(f"Required config missing for model {modelCode} : {rhml_configreader.MODEL_RF_NUM_FEATURES} must be > 0")
            return None
        myRandomForest = RegressionRandomForest.RHMLRegressionRandomForest(numberFeatures,numberTrees,data,labels,**modelKwargs)
        return myRandomForest

    elif modelCode == rhml_configreader.MODEL_BOOST:
        boostedModel = RegressionGradBooster.RHMLRegressionGradBooster(numberTrees,data,labels,**modelKwargs) #{"min_split":25,"max_depth":6}
        return boostedModel
    return None


# Assumes we have got data split into train and test : 
def runTests(theModel,X_train, X_test, y_train, y_test):

    modelSummary = theModel.getSummary()

    # get feature importance
    featureImp = theModel.calcFeatureImportance()

    # run the training data through : 
    training_results = theModel.test(X_train,y_train)

    # run through the test data:
    testing_results = theModel.test(X_test,y_test)

    # OOB results may be available too ... if not set it to None 
    modelOOBResults = getattr(theModel,"OOBScore",None)

    return modelSummary,featureImp,training_results,testing_results,modelOOBResults



# Enforce any required defaults here
def amendForDefaults(problemType,modelCode,modelKwargs):
    if problemType == rhml_dataloader.CLASSIFICATION:
        if modelCode == rhml_configreader.MODEL_BOOST:
            if not modelKwargs.get('max_depth'):
                modelKwargs['max_depth'] = AdaBooster.RHMLAdaBooster.DEFAULT_MAX_DEPTH


# If the underlying model supports the calcProximity method, call it and return results
def calcProximityMatrix(theModel,X_train):
    proximityMatrix = None
    modelProximityMatrixMethod = getattr(theModel,"calcProximity",None)
    if modelProximityMatrixMethod is not None:
        print("Model supports proximity")
        proximityMatrix = theModel.calcProximity(X_train)
    else:
        print("Model DOES NOT support Proximity")

    return proximityMatrix


# For cross validation, create a set of folds each of which contain some random indexs
def createCVFoldDataIndexes(numberOfDataItems,numberOfFolds):
    # we want some random indexs in each of the folds, but only one index in each fold
    numberPerFold = int(math.ceil(numberOfDataItems/numberOfFolds))
    allIndexes = list(range(numberOfDataItems))
    random.shuffle(allIndexes)

    allFolds=[]
    for i in range(numberOfFolds):
        theFold = allIndexes[(i*numberPerFold):(((i+1)*numberPerFold))-1]
        allFolds.append(theFold)

    return allFolds

# For a given cross validation iteration, work out the indexs to be used for training and for test
# for example, for iteration 3, you would keep the 3rd one for test and merge the others for training
def getTrainTestIndexesForCVIteration(iteration, folds):
    mergedTraining =[]
    for i,fold in enumerate(folds):
        if i != iteration:
            mergedTraining.extend(fold)
    
    return mergedTraining,folds[iteration]

# getListParams
# Find paramaters that have a list form ie use sqaure brqaces [ ]
# This returns the key names in one array and the list of values in a separate array
def getListParams(modelKwargs):
    paramNames = []
    paramValues = []
    for key,value in modelKwargs.items():
        if isinstance(value,List):
            print(f"We have a range for : {key} and it looks like this  {value}")
            paramNames.append(key)
            paramValues.append(value)
    return paramNames,paramValues

# Get a seed value - uses default but can be set in config too
def getRandomStateSeed(configReader):
    random_state_seed=101
    config_seed = rhml_configreader.getRandomSeedFromConfig(configReader)
    if config_seed != "":
        random_state_seed = int(config_seed)
    return random_state_seed

    
