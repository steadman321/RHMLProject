###############################################################################################
# Class Name    :       test_RHMLDecisionTree
# Description   :       Test the Base Descision Tree Implementation 
################################################################################################

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
from RHML.Utils.Plots import plotFeatureImportance


def test_regression():
    # first create some data (regression)
    x_train, y_train = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # now make up some test observations
    testObservations = []
    testObservations.append(np.array([0,-2.5])) #bottom
    testObservations.append(np.array([0,4])) #top 
    testObservations.append(np.array([0,0])) #mid
    testObservations.append(np.array([0,-2.5])) #bottom
    testObservations.append(np.array([0,4])) #top 
    testObservations.append(np.array([0,0])) #mid
    
    # now fir data to a regression tree
    from RHML.DTrees import RegressionTree as RTree 
    rtree = RTree.RHMLRegressionDecisionTree(x_train,y_train)
    rtree.describe()
    featureImp=rtree.calcFeatureImportance()

    # now get some predictions ......... 
    results = rtree.predict(testObservations)
    print(f"Predictions look like this  {results}")

    show_regression(x_train,y_train,testObservations,results)
    plotFeatureImportance([featureImp])

def test_classification():
    # first create some data (classification)
    x_train, y_train = make_blobs(n_samples=100,n_features=2, centers=3, random_state=101)
    
    # now make up some test observations
    testObservations = []
    testObservations.append(np.array([-10,-6])) #class1
    testObservations.append(np.array([4,7.5]))  #class2
    testObservations.append(np.array([0,1]))    #class0
    # testObservations.append(np.array([-5,10]))

    # now try classification tree
    from RHML.DTrees import ClassificationTree as CTree 
    ctree = CTree.RHMLClassificationDecisionTree(x_train,y_train)
    ctree.describe()
    featureImp = ctree.calcFeatureImportance()

    results = ctree.predict(testObservations)
    print(f"Predictions (ctree) is {results}")

    # also work out probs for each class for each test observation 
    class0, probs0 = ctree.calcClassProbabilities(testObservations[0])
    class1, probs1 = ctree.calcClassProbabilities(testObservations[1])
    class2, probs2 = ctree.calcClassProbabilities(testObservations[2])
    print("Class probabilities")
    print(f"Obs0 : classes {class0}, probs: {probs0}")
    print(f"Obs1 : classes {class1}, probs: {probs1}")
    print(f"Obs2 : classes {class2}, probs: {probs2}")

    show_classification(x_train,y_train,testObservations,results)
    plotFeatureImportance([featureImp])

def show_regression(X1,Y1,observations,predictedValues):
    plt.figure(figsize=(8, 8))
    plt.title("Regression Test", fontsize='small')

    # observatoinScatterX= np.array([[observation[0],observation[1]]])
    
    # amend test data to have a col for size scaling = 1
    colOfOnes = np.ones((X1.shape[0],1))
    X1 = np.append(X1,colOfOnes,axis=1)
    
    # add in our test point into the data (with scale set to 10)
    observations = np.array(observations)
    colOfTens = (np.ones((observations.shape[0],1)))*10
    
    observations= np.append(observations,colOfTens,axis=1)  
    # observationScatterY = np.array([predictedValue])
    predictedValues = np.array(predictedValues)

    X1 = np.vstack((X1,observations))
    Y1 = np.append(Y1,predictedValues)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,s=25*X1[:,2], edgecolor='k')

    plt.show()

def show_classification(X1,Y1,observations,predictedValues):
    plt.figure(figsize=(8, 8))
    classes=['0','1','2','4']

    observations= np.array(observations)
    predictedValues = np.array(predictedValues)

    colormap = np.array(['r', 'g', 'b','y'])

    # # plot in our test point
    plt.scatter(observations[:,0], observations[:,1], marker='D',s=250, edgecolor='k',c=colormap[predictedValues])

    # add in the test data 
    plt.scatter(X1[:, 0], X1[:, 1], marker='o',s=25, edgecolor='k',c=colormap[Y1])

    # cook up a legend
    from matplotlib.lines import Line2D
    legend_elements = [ 
                        Line2D([0], [0], marker='o', color=colormap[0], label='Class '+classes[0], markersize=5, linewidth=0.0),
                        Line2D([0], [0], marker='o', color=colormap[1], label='Class '+classes[1], markersize=5, linewidth=0.0),
                        Line2D([0], [0], marker='o', color=colormap[2], label='Class '+classes[2], markersize=5, linewidth=0.0),
                    ]

    plt.legend(handles=legend_elements)

    plt.show()



def test_regression_Performance():
    # first create some data (regression)
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1,random_state=101)
    
    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 
   
    # now fit data to a regression tree
    from RHML.DTrees import RegressionTree as RTree 
    rtree = RTree.RHMLRegressionDecisionTree(X_train,y_train,**{"min_split":25,"max_depth":9})
    rtree.describe()
    featureImp = rtree.calcFeatureImportance()

    # Lets get some stats for the training data first! 
    training_results = rtree.test(X_train,y_train)
    print(f"Training data :")
    print(f"MSE: {training_results.score[1]}")
    print(f"R2: {training_results.score[0]}")

    # now get some predictions ......... i need some actual labels too!!!! oops!
    results = rtree.test(X_test,y_test)
    print(f"test results looks like this :")
    print(f"MSE: {results.score[1]}")
    print(f"R2: {results.score[0]}")

    plotFeatureImportance([featureImp])


def test_classification_Performance():
    # first create some data (classification)
    X, y = make_blobs(n_samples=1000,n_features=2, centers=3, random_state=101)

    # Split data into test and train - we'll take a third as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 
   

    # now create (and fit) classification tree
    from RHML.DTrees import ClassificationTree as CTree 
    ctree = CTree.RHMLClassificationDecisionTree(X_train,y_train,**{"min_split":150})
    # ctree = CTree.RHMLClassificationDecisionTree("ClassTreeData",X_train,y_train)

    ctree.describe()
    featureImp = ctree.calcFeatureImportance()

   # Lets get some stats for the training data first! 
    training_results = ctree.test(X_train,y_train)
    print(f"Training data :")
    print(f"measure1: {training_results.score[0]}")
    print(f"measure2: ")
    cmatrix = training_results.score[1]
    print(f"{cmatrix.getConfusionMatrix()}")
    # ctree.calcConfusionMatrixMetrics(training_results.score[1][0],training_results.score[1][1])
    # cmatrix.calcConfusionMatrixMetrics()
    metrics = cmatrix.getMetrics()
    for i,metric in enumerate(metrics):
        print(f"Metric {i} : TP:{metric.TP} FP:{metric.FP} FN:{metric.FN} TN:{metric.TN} AC:{metric.AC} PR:{metric.PR} RC:{metric.RC} F1:{metric.F1}")

    # show confusion matrix 
    test_results = ctree.test(X_test,y_test)
    print(f"test results looks like this :")
    print(f"measure1: {test_results.score[0]}")
    print(f"measure2: ")
    cmatrix = test_results.score[1]
    print(f"{cmatrix.getConfusionMatrix()}")
    # ctree.calcConfusionMatrixMetrics(results.score[1][0],results.score[1][1])
    # cmatrix.calcConfusionMatrixMetrics()
    metrics = cmatrix.getMetrics()
    for i,metric in enumerate(metrics):
        print(f"Metric {i} : TP:{metric.TP} FP:{metric.FP} FN:{metric.FN} TN:{metric.TN} AC:{metric.AC} PR:{metric.PR} RC:{metric.RC} F1:{metric.F1}")

    # this plot just plots out the training and test data - but uses the expected test data labels - I want to see the labels I predict - as not all will be correct! 
    # how do I get y_test replaced with my test results here ? my test will return .predictions
    # show_classification(X_train,y_train,X_test,y_test)
    # this works 
    # show_classification(X_train,y_train,X_test,test_results.predictions)
    
    # now how baout showing the test points as actuals V test points as predicted! ie not show train at all!This is good, shows one misclass if have data =1000
    show_classification(X_test,y_test,X_test,test_results.predictions)
    plotFeatureImportance([featureImp])

# Rather than get prediction, just get node index for each observation
def test_classification_nodes():
    # first create some data (classification)
    x_train, y_train = make_blobs(n_samples=100,n_features=2, centers=3, random_state=101)

    # now try classification tree
    from RHML.DTrees import ClassificationTree as CTree 
    ctree = CTree.RHMLClassificationDecisionTree(x_train,y_train)
    ctree.describe()

    nodeIndexes = ctree.calcNodes(x_train)
    print(f"Nodes (ctree) are {nodeIndexes}")


# Run some tests : 
# test_regression()
# test_classification()
# test_regression_Performance()
test_classification_Performance()
# test_classification_nodes()
#"split_fearture_indexes":[0,1]