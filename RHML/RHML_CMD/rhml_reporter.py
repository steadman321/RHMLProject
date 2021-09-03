################################################################################################
# rhml_reporter
# 
# This will create a report based on RHML model results and open this up in the default browser. 
# The 3 main report styles suported currently are : 
# 
###############################################################################################
import webbrowser
import os,sys
import numpy as np
import math
from RHML.Utils.Plots import saveFeatureImportancePlot,plotAccuracy,plotGridResults,plotGridHeatResults,plotMultiModel
from RHML.Utils.Plots import plotProximityClassification,plotProximityRegresion

import rhml_dataloader

################################################################################################
# buildReport : 
# The main 'single split' report. 
# Supports classification and regression reports 
# One report per model
################################################################################################
def buildReport(dataName, problemType,title,desc,featureList,featuresToSkip,modelKwargs,modelSummary,
                                featureImp,training_results,testing_results,proximityMatrix,actualLables,modelOOBResults):

    # This will build reports for classifcattion and regression , but they are slightly differnt in format!
    # However,lots in commmon too!So just pull out the differnece along the way
    doingClassification = True
    if problemType != rhml_dataloader.CLASSIFICATION:
        doingClassification = False

    #setup some file name nad paths 
    module_dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = module_dir_path+'/reports/'+title+'_report.html'
    featureImpPNG = module_dir_path+'/reports/figs/'+title+'_featureImp.png'
    trainingAccPNG = module_dir_path+'/reports/figs/'+title+'_trainingAcc.png'
    testingAccPNG = module_dir_path+'/reports/figs/'+title+'_testingAcc.png'
    oobAccPNG = module_dir_path+'/reports/figs/'+title+'_oobAcc.png'
    proximityPNG = module_dir_path+'/reports/figs/'+title+'_proximity.png'
    dataOverviewFilename = module_dir_path+'/data/'+dataName+'/'+dataName+'.html'
    
    # add some details on the data used 
    dataDetailsHtml = getDataDetailsHtml(dataOverviewFilename)

    import urllib.parse
    filelink = 'file:///'+urllib.parse.quote(filename)

    # read in base template
    if doingClassification:
        htmlTemplate = getTemplateAsString("classificationReportTemplate.html")
    else:
        htmlTemplate = getTemplateAsString("regressionReportTemplate.html")

    # read in css style section from templates too (injected into html header!)
    styleSection=getTemplateAsString("styles.html")

    # summary should contain tree structure
    nodeData = modelSummary
    
    # from resutls we want to extract metrics from [0] and confusion matrix from [1]
    training_score =training_results.score[0]
    training_score_percent = "{:.2%}".format(training_score)
    test_score = testing_results.score[0]
    test_score_percent = "{:.2%}".format(test_score)

    if doingClassification:
        allClasses = training_results.score[1].getAllClasses()
        trainingCMatrix = training_results.score[1].getConfusionMatrix()
        trainingCMatrixMetrics = training_results.score[1].getMetricsSummary()
        testingCMatrix = testing_results.score[1].getConfusionMatrix()
        testingCMatrixMetrics = testing_results.score[1].getMetricsSummary()
        trainingCMatrixSection = createCFMatrixHtml(trainingCMatrix,allClasses,trainingCMatrixMetrics)
        trainingAccSection = "<b>"+str(training_score_percent)+"</b>"
        testinCMatrixSection = createCFMatrixHtml(testingCMatrix,allClasses,testingCMatrixMetrics)
        testingAccSection = "<b>"+str(test_score_percent)+"</b>"
    
    else:
        training_score_MSE = training_results.score[1]
        training_score_RMSE = math.sqrt(training_score_MSE)
        test_score_MSE = testing_results.score[1]
        test_score_RMSE = math.sqrt(test_score_MSE)
        trainingR2Section = str(training_score_percent)
        trainingMSESection = str(training_score_MSE)
        trainingRMSESection = str(training_score_RMSE)

        testingR2Section = str(test_score_percent)
        testingMSESection = str(test_score_MSE)
        testingRMSESection = str(test_score_RMSE)


    # build up node data - this is for tree structure section
    nodeSection = createNodeHtml(nodeData)

    showTreeStructure = ""
    if len(nodeData)==0:
        showTreeStructure = "none"

    # add some details about any features skipped (could be none , in which case hide this bit!)
    showSkips=""
    if len(featuresToSkip)==0:
        showSkips="none"
        skipinfo=""
    else:
        skipinfo=str(featuresToSkip).upper().strip('[]').replace("'","")


    # Create a feature importance plot using matplotlib and save it to disk; this will then be included in the report
    saveFeatureImportancePlot([featureImp],featureImpPNG)

    # create plot for training acc, and test acc
    if doingClassification:
        plotAccuracy(training_score,trainingAccPNG)
        plotAccuracy(test_score,testingAccPNG)
    else:
        plotAccuracy(training_score,trainingAccPNG,label1="Variation Explained",label2="")
        plotAccuracy(test_score,testingAccPNG,label1="Variation Explained",label2="")

    # may have OOB numbers too - this should only be done if needed 
    showOOB="none"
    hideForOOB=""
    oobCMatrixSection=""
    oobAccSection=""
    oobAccSection=""
    oobMSESection=""
    oobRMSESection=""
    if modelOOBResults is not None:
        showOOB=""
        hideForOOB="none"
        if doingClassification:
            oob_score = modelOOBResults[0]
            oob_score_percent = "{:.2%}".format(oob_score)
            oobCMatrix = modelOOBResults[1].getConfusionMatrix()
            oobCMatrixMetrics = modelOOBResults[1].getMetricsSummary()
            plotAccuracy(oob_score,oobAccPNG)
            allClasses = modelOOBResults[1].getAllClasses()
            oobCMatrixSection = createCFMatrixHtml(oobCMatrix,allClasses,oobCMatrixMetrics)
            oobAccSection = "<b>"+str(oob_score_percent)+"</b>"
        else:
            oob_r2_score = modelOOBResults[0]
            oob_r2_score_percent = "{:.2%}".format(oob_r2_score)
            plotAccuracy(oob_r2_score,oobAccPNG,label1="Variation Explained",label2="")
            oobAccSection = str(oob_r2_score_percent)
            oobMSESection = str(modelOOBResults[1])
            oobRMSESection = str(math.sqrt(modelOOBResults[1]))
    else:
        oobAccPNG=""


    # get the kwargs into a html format to display too:
    modelHyperparams = getHyperParamHtml(modelKwargs)

    # build up the feature list html 
    featureListHtml = getFeatureListHtml(featureList)

    # get a list of features ordered by importance
    sort_index = np.argsort(featureImp)
    sort_reverse = sort_index[::-1]
    orderedFeatures = featureList[sort_reverse]
    # reverseOrderedFeatures = orderedFeatures[::-1]
    orderedFeaturesHtml = getOrderedFeatureListHtml(sort_reverse, orderedFeatures)

    # If we have a proximity matrix show that too :
    if proximityMatrix is not None:
        if doingClassification:
            plotProximityClassification(proximityMatrix,actualLables,proximityPNG)
        else:
            plotProximityRegresion(proximityMatrix,proximityPNG)
        showProximity=""
    else:
        proximityPNG=""
        showProximity="none"

    reportValues = {
        "TITLE":title,
        "DESC":desc,
        "HYPERPARAMS":modelHyperparams,
        "SHOW_TREE_STRUCTURE":showTreeStructure,
        "NODEINFO":nodeSection,
        "STYLE":styleSection,
        "FEATUREIMP_PNG":featureImpPNG,
        "TRAININGACC_PNG":trainingAccPNG,
        "TESTINGACC_PNG":testingAccPNG,
        "FEATURELIST":featureListHtml,
        "ORDEREDFEATURES":orderedFeaturesHtml,
        "DATADETAILS":dataDetailsHtml,
        "SHOW_SKIPS":showSkips,
        "SKIPPED_FEATURE_DETAILS":skipinfo,
        "PROXIMITY_PNG":proximityPNG,
        "OOBACC_PNG":oobAccPNG,
        "OOB_CMATRIX":oobCMatrixSection,
        "OOB_ACC":oobAccSection,
        "SHOW_OOB":showOOB,
        "SHOW_PROX":showProximity,
        "HIDE_FOR_OOB":hideForOOB

    }
    # add in classification specfiics
    if doingClassification:
        classificationReportSettings = {
            "TRAIN_CMATRIX":trainingCMatrixSection,
            "TRAIN_ACC":trainingAccSection,
            "TEST_CMATRIX":testinCMatrixSection,
            "TEST_ACC":testingAccSection
        }
        reportValues.update(classificationReportSettings)
    else:
        regressionReportSettings = {
            "TRAIN_R2":trainingR2Section,
            "TRAIN_MSE":trainingMSESection,
            "TRAIN_RMSE":trainingRMSESection,
            "TEST_R2":testingR2Section,
            "TEST_MSE":testingMSESection,
            "TEST_RMSE":testingRMSESection,
            "OOB_MSE":oobMSESection,
            "OOB_RMSE":oobRMSESection
        }
        reportValues.update(regressionReportSettings)

    formatedHTML = htmlTemplate.format(**reportValues)

    # now write it to disk
    with open(filename,'w+') as writer:
        writer.write(formatedHTML)

    print(f"report filename looks like this {filelink}")
    webbrowser.open_new_tab(filelink)

################################################################################################
# buildGridSearchReport : 
# A single report per model, with multiple moving paramters 
# If there is one moving paramter, a simple graph will be included
# If there are 2 moving paramters, a heat map will be included 
# If there are more than 2 moving paramters then just the raww results will be shown in a table
################################################################################################
def buildGridSearchReport( dataName,featuresToSkip,title,desc,searchParamGrid,grid,grid_results,best_index,modelKwargs):

    #setup some file name and paths 
    module_dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = module_dir_path+'/reports/'+title+'_report.html'
    gridSearchPlotPNG = module_dir_path+'/reports/figs/'+title+'_gridSearchPlot.png'
    import urllib.parse
    filelink = 'file:///'+urllib.parse.quote(filename)
    dataOverviewFilename = module_dir_path+'/data/'+dataName+'/'+dataName+'.html'
    
    # add some details on the data used 
    dataDetailsHtml = getDataDetailsHtml(dataOverviewFilename)

    # read in base template
    htmlTemplate = getTemplateAsString("gridSearchReportTemplate.html")

    # read in css style section from templates too 9injected into html header!)
    styleSection=getTemplateAsString("styles.html")

    # get the kwargs into a html format to display too:
    modelHyperparams = getHyperParamHtml(modelKwargs)

    # get the grid results html 
    gridResultsHtml = getGridSearchResultsTableHtml(grid,grid_results)

    # Do a plot if we have one or two params to show
    numParams = len(searchParamGrid)
    if numParams==1:
        gridResultsPlotHtml = getGridSearchResultsPlotHtml(grid,grid_results,gridSearchPlotPNG)
    elif numParams ==2: 
        gridResultsPlotHtml = getGridSearchResultsHeatPlotHtml(searchParamGrid,grid,grid_results,gridSearchPlotPNG)
    else:
        gridSearchPlotPNG=""

    # add some details about any features skipped (could be none , in which case hide this bit!)
    showSkips=""
    if len(featuresToSkip)==0:
        showSkips="none"
        skipinfo=""
    else:
        skipinfo=str(featuresToSkip).upper().strip('[]').replace("'","")



    # values to put into report
    reportValues = {
        "TITLE":title,
        "DESC":desc,
        "STYLE":styleSection,
        "HYPERPARAMS":modelHyperparams,
        "GRIDRESULTSTABLE":gridResultsHtml,
        "GRIDRESULTS_PNG":gridSearchPlotPNG,
        "DATADETAILS":dataDetailsHtml,
        "SHOW_SKIPS":showSkips,
        "SKIPPED_FEATURE_DETAILS":skipinfo
    }
    formatedHTML = htmlTemplate.format(**reportValues)

    # now write it to disk
    with open(filename,'w+') as writer:
        writer.write(formatedHTML)


    print(f"report filename looks like this {filelink}")
    webbrowser.open_new_tab(filelink)

################################################################################################
# buildMultiModelReport : 
# A single report covering multipl models and one moving paramter
################################################################################################
def buildMultiModelReport( dataName,featuresToSkip,reportTitle,reportDesc, models_run,model_kwargs
                            ,movingParameters,movingParamaterValues,training_results_cache
                            ,testing_results_cache,oob_results_cache):

    #setup some file name nad paths 
    module_dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = module_dir_path+'/reports/multimodel/'+reportTitle+'_report.html'
    plotPNG = module_dir_path+'/reports/multimodel/figs/'+reportTitle+'_plot.png'

    import urllib.parse
    filelink = 'file:///'+urllib.parse.quote(filename)

    # read in base template
    htmlTemplate = getTemplateAsString("multiModelReportTemplate.html")

    # read in css style section from templates too (injected into html header!)
    styleSection=getTemplateAsString("styles.html")

    # Create a plot for the results - this is the main part of the report  
    plotMultiModel(models_run,movingParameters,movingParamaterValues,training_results_cache,testing_results_cache,oob_results_cache,plotPNG)

    dataOverviewFilename = module_dir_path+'/data/'+dataName+'/'+dataName+'.html'
    
    # add some details on the data used 
    dataDetailsHtml = getDataDetailsHtml(dataOverviewFilename)
    # add some details about any features skipped (could be none , in which case hide this bit!)
    showSkips=""
    if len(featuresToSkip)==0:
        showSkips="none"
        skipinfo=""
    else:
        skipinfo=str(featuresToSkip).upper().strip('[]').replace("'","")

    # for settings, we need details of settings for each model run 
    # models_run,model_kwargs
    modelHyperparams = ""

    for modelName,modelKwargs in zip(models_run,model_kwargs):
        modelHyperparams+="<b>Model:"+modelName.upper()+"</b></p>"
        if len(modelKwargs)>0:
            modelHyperparams+= getHyperParamHtml(modelKwargs)
        else:
            modelHyperparams+="(System defaults only)</p>"

    reportValues = {
        "TITLE":reportTitle,
        "DESC":reportDesc,
        "STYLE":styleSection,
        "PLOT_PNG":plotPNG,
        "DATADETAILS":dataDetailsHtml,
        "SHOW_SKIPS":showSkips,
        "SKIPPED_FEATURE_DETAILS":skipinfo,
        "HYPERPARAMS":modelHyperparams,
    }

    formatedHTML = htmlTemplate.format(**reportValues)

    # now write it to disk
    with open(filename,'w+') as writer:
        writer.write(formatedHTML)

    print(f"report filename looks like this {filelink}")
    webbrowser.open_new_tab(filelink)

################################################################################################
# Supporting functions to create specific html sections and create plots etc
################################################################################################
def createNodeHtml(nodeData):
    # read in node table template
    nodeTableTemplate = getTemplateAsString("nodeTableTemplate.html")

    # build up some html for the rows oof data.....
    nodeSectionHtml=""
    for node in nodeData:
        nodeSectionHtml+="<tr>"
        for nodeField in node:
            nodeSectionHtml+= "<td>"+nodeField+"</td>"
        nodeSectionHtml+="</tr>"

    # inject our rows of data into the template
    injectedValues = {
        "NODEROWS":nodeSectionHtml
    }
    finalHTML = nodeTableTemplate.format(**injectedValues)

    return finalHTML

def createCFMatrixHtml(cmatrixData,allClasses,cmatrixMetrics):
    if cmatrixData is None:
        return ""

    # read in template with placeholders for our dynamic content 
    cmatrixTemplate = getTemplateAsString("confusionMatrixTemplate.html")

    # new top top row = label for "predicted class" label 
    cfm_headers_html = ""
    for c in allClasses:
        cfm_headers_html+="<td class='cmatrix_header'>"+str(c)+"</td>"

    cfm_data1_html=""
    # add in data for confusion matrix
    for row in range(cmatrixData.shape[0]):
        cfm_data1_html+="<tr><td class='cmatrix_header_side'>"+str(allClasses[row])+"</td>"
        for col in range(cmatrixData.shape[1]):
            val = str(cmatrixData[row,col])
            classname=""
            if row == col:
                classname="confMatrixDiag"
            cfm_data1_html+= "<td class='"  +  classname  +    "'>"+val+"</td>"
        cfm_data1_html+="</tr>"

    # add in the metrics data - first for the classes (which is all but last 3 rows of the summary)
    cfm_data2_html=""
    classRows = (len(cmatrixMetrics))-3
    for row in range(classRows):
        cfm_data2_html+="<tr><td class='cmatrix_header_side'>"+str(allClasses[row])+"</td>"
        for col in range(len(cmatrixMetrics[row])):
            val = format(cmatrixMetrics[row][col],".2f")
            cfm_data2_html+= "<td>"+val+"</td>"
        cfm_data2_html+="</tr>"

    # add in data for the other metrics data (last 3 rows of summary)
    metricNames=["Micro", "Macro", "Weighted"]
    for i,header in enumerate(metricNames):
        cfm_data2_html+="<tr><td class='cmatrix_header_side'>"+header+"</td>"
        theRow = (len(cmatrixMetrics)-3)+i
        for col in range(len(cmatrixMetrics[row])):
            val = format(cmatrixMetrics[theRow][col],".2f")
            cfm_data2_html+= "<td>"+val+"</td>"
        cfm_data2_html+="</tr>"

    # now push dynamic content into the html template
    injectedValues = {
        "CFM_HEADERS":cfm_headers_html,
        "CFM_DATA1":cfm_data1_html,
        "CFM_DATA2":cfm_data2_html
    }
    finalHTML = cmatrixTemplate.format(**injectedValues)

    return finalHTML

# Read in a template file as a string. 
# These templates will have tokens in {} brances which will get replaced with data
def getTemplateAsString(filename):
    module_dir_path = os.path.dirname(os.path.realpath(__file__))
    filelocation = module_dir_path+'/templates/'+filename
    with open(filelocation, 'r') as fileReader:
        fileAsString = fileReader.read().replace('\n', '')
    return fileAsString

# Get a static description of what the hyperparams are used for 
def getHyperParamDescription(paramCode):

    # if the param code has got :  [code default] includes , strip it!
    # infact strip anything after first space including first space
    # paramCode = paramCode.rstrip('[')
    paramCode = paramCode.split(' ')[0]
    
    paramDescriptions = dict()

    paramDescriptions['min_split']="The minimum number of data points in a region to allow a split"
    paramDescriptions['max_depth']="The maximum level of splits allowed in a decision tree"
    paramDescriptions['include_oob']="Calculate the out-of-bag scores [if applicable]"
    paramDescriptions['num_features']="The size of the subset of features used for each base model"
    paramDescriptions['num_trees']="The number of base models (Decision Trees) used in the ensemble"
    paramDescriptions['ada_multi_class']="Set to 1 to use the multi-class version of the AdaBoost algorithm"

    return paramDescriptions.get(paramCode, "")

# Core code defaults : could get these from the core tree class instead?
CORE_DEFAULT_MIN_SPLIT=25
CORE_DEFAULT_MAX_DEPTH=1000

# adjustKwargsForCodeDefaults
# Special Case for code -defaults 
# If no kwargs are set in config there are still some defaults in the base model code 
# If these are being used we should include these too and mark them as code defaults (to dinstinquish from ALL_MODEL defaults)
def adjustKwargsForCodeDefaults(modelKwargs):
    adjustedKwargs = modelKwargs.copy()
    if adjustedKwargs.get('min_split', None) is None:
        adjustedKwargs['min_split [code default]']=CORE_DEFAULT_MIN_SPLIT
    
    if adjustedKwargs.get('max_depth', None) is None:
        adjustedKwargs['max_depth [code default]']=CORE_DEFAULT_MAX_DEPTH
    
    return adjustedKwargs

def getHyperParamHtml(modelKwargs):
    # import json
    # return json.dumps(modelKwargs)
    hyperTableTemplate = getTemplateAsString("hyperparamsTableTemplate.html")
    paramsRowsHtml=""

    # Special Case for code -defaults 
    modelKwargs = adjustKwargsForCodeDefaults(modelKwargs)
    
    # data is np array : want it as html table here
    for key in modelKwargs:
        paramsRowsHtml+="<tr><td>"+key+"</td><td>"+str(modelKwargs[key])+"</td><td>"+getHyperParamDescription(key)+"</td></tr>"

    # inject our rows of data into the template
    injectedValues = {
        "HYPERPARMROWS":paramsRowsHtml
    }
    finalHTML = hyperTableTemplate.format(**injectedValues)
    
    return finalHTML

def getFeatureListHtml(featureList):
    feautureHtml= "<div><ol start=0>"

    for feature in featureList:
        feautureHtml+="<li>"+feature+"</li>"
    feautureHtml= feautureHtml+"</ol></div>"
    
    return feautureHtml

def getOrderedFeatureListHtml(sort_reverse, orderedFeatures):
    feautureHtml= "<div><ol start=0>"

    for index,feature in enumerate(orderedFeatures):
        feautureHtml+='<li value="'+str(sort_reverse[index])+'">'+feature+'</li>'
    feautureHtml= feautureHtml+"</ol></div>"
    
    return feautureHtml

def getGridSearchResultsTableHtml(grid,gridResults):
    # I want to sort things : get the index of a sorted version of gridResults 
    # as want to sort things from both grid and gridResults in that same order
    sortedIndexes = np.argsort(gridResults)
    revSortedIndexes = sortedIndexes[::-1][:len(sortedIndexes)]
    
    scoreRowsHtml=""
    for i in revSortedIndexes:
        gridSettingsText = str(grid[i]).strip('{}').replace("'","")
        scoreRowsHtml+="<tr><td>"+gridSettingsText+"</td><td>"+str(gridResults[i])+"</td></tr>"

    # Get the html template to house the results 
    gridResultsTableTemplate = getTemplateAsString("gridsearchResultsTableTemplate.html")
    
    # inject our rows of data into the template
    injectedValues = {
        "GRIDRESULT_ROWS":scoreRowsHtml
    }
    finalHTML = gridResultsTableTemplate.format(**injectedValues)
    
    return finalHTML

# This is only used if we have a single grid-search paramter 
# This shows a simple line graph for the score V tested params values
def getGridSearchResultsPlotHtml(grid,gridResults,filename):
    # Re-org the results into something we can use ... 
    X = []
    Y= []

    for i in range(len(gridResults)):
        gridValue= list(grid[i].values())[0]
        paramLabel = list(grid[i].keys())[0]
        X.append(gridValue)
        Y.append(gridResults[i])

    # sort on X value 
    X.sort()
    Y_sorted = [x for _,x in sorted(zip(X,Y))]
    plotGridResults(X,Y_sorted,filename,label1=paramLabel)

# This is used where we have 2 grid-search params. 
# Results are presented as a heat-matrix
def getGridSearchResultsHeatPlotHtml(searchParamGrid,grid,gridResults,filename):

    axis_labels=[]
    data_point_labels=[]
    
    for key in searchParamGrid.keys():
        axis_labels.append(key)
        data_point_labels.append(searchParamGrid[key])

    # Assuming only 2 params
    x_axis_label = axis_labels[0]
    x_labels = data_point_labels[0]
    y_axis_label = axis_labels[1]
    y_labels = data_point_labels[1]

    # now create an empty matrix for our heatmap
    lenX = len(x_labels)
    lenY = len(y_labels)

    # NOTE: switch x and y : numpy has x = num rows, and i'm using y for that here!
    heatmap_data = np.zeros((lenY,lenX))

    # the order of results (inside gridResults) is always based on the alpha order of the search params keys ... 
    # this impacts the order we fetch things and add to the plot since we plot by order seen in config file 
    # hence need to make amendments here to get ordering to match
    firstLen= lenY
    secondLen = lenX
    across=True
    if min(x_axis_label,y_axis_label)!=y_axis_label:
        firstLen= lenX
        secondLen = lenY
        across=False

    # populate the heatmap data
    d_index = 0
    for i in range(firstLen):
        for j in range(secondLen):
            if across:
                heatmap_data[i,j]=gridResults[d_index]
            else:
                heatmap_data[j,i]=gridResults[d_index]
            d_index=d_index+1

    # now have enough to do the actual plot 
    plotGridHeatResults(heatmap_data,filename,x_axis_label,y_axis_label,x_labels,y_labels)


def getDataDetailsHtml(dataOverviewFilename):
    with open(dataOverviewFilename, 'r') as fileReader:
        fileAsString = fileReader.read().replace('\n', '')
    return fileAsString
    








    