###############################################################################################
# Plot.py
# Functions to support plotting various graphs for the reporter module
###############################################################################################
import matplotlib.pyplot as plt
import numpy as np

def plotFeatureImportance(featureImps):
    numberPlots = len(featureImps)
    fig,axs = plt.subplots(numberPlots)

    for axis in range(numberPlots):
        featureData= featureImps[axis]
        y_pos = np.arange(len(featureData))
        # plt.figure()
        # fig.suptitle("Feature importances")
        if numberPlots == 1:
            axs.set_title('Feature importances')
            axs.bar(y_pos, featureData,
            color="r", align="center")
        else:
            axs[axis].set_title('Feature importances')
            axs[axis].bar(y_pos, featureData,
            color="r", align="center")
    
    plt.show()

def saveFeatureImportancePlot(featureImps,filename):
    numberPlots = len(featureImps)
    fig,axs = plt.subplots(numberPlots)

    for axis in range(numberPlots):
        featureData= featureImps[axis]

        # alternate colors - add enough for the number of features we are dealing with
        color1='#17becf'
        color2='#1f77b4'
        colors=[color1]
        for i in range(len(featureData)):
            if colors[-1]==color1:
                colors.append(color2)
            else:
                colors.append(color1)

        y_pos = np.arange(len(featureData))
        if numberPlots == 1:
            axs.bar(y_pos, featureData,
            color=colors, align="center")
        else:
            axs[axis].bar(y_pos, featureData,
            color="r", align="center")
    
    plt.savefig(filename,bbox_inches='tight')

def plotAccuracy(accuracy_percentage,filename,label1="Correct",label2="Wrong"):
    fig = plt.figure()
    fig.set_size_inches(2,2)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    pielabels=[label1]
    results = []
    results.append(accuracy_percentage)
    if accuracy_percentage<1:
        results.append(1-accuracy_percentage)   
        pielabels.append(label2)
        ax.pie(results, labels = pielabels,explode=(0,0.25))
    else:
        ax.pie(results, labels = pielabels,)
    plt.savefig(filename,bbox_inches='tight')

def plotGridResults(X,Y,filename,label1="Parameter Values",label2="Score"):
    fig = plt.figure()
    ax=plt.axes()
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    ax.plot(X,Y)
    plt.savefig(filename,bbox_inches='tight')

def plotGridHeatResults(heatmap_data,filename,x_axis_label,y_axis_label,x_labels,y_labels):
    fig = plt.figure()
    ax=plt.axes()
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    img = ax.imshow(heatmap_data,cmap='hot')#,extent=[x_labels[0],x_labels[-1],y_labels[0],y_labels[-1]]
    
    # Set up ticks and labels to be our grid settings (default is just linear)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)

    fig.colorbar(img)
    plt.savefig(filename,bbox_inches='tight')

def plotProximityRegresion(proximityMatrix,filename):
    fig = plt.figure()
    from sklearn.manifold import MDS
    mds = MDS(2,random_state=0,dissimilarity='precomputed')
    X_2d = mds.fit_transform(proximityMatrix)
    # plt.rcParams['figure.figsize'] = [7, 7]
    # plt.rc('font', size=14)
    x = [row[0] for row in X_2d]
    y = [row[1] for row in X_2d]
    plt.scatter(x,y)
    # plt.legend()
    # plt.show()
    plt.savefig(filename,bbox_inches='tight')

# This version gets given unique lables! could break tests here!
def plotProximityClassification(proximityMatrix,actualLables,filename):
    fig = plt.figure()
    from sklearn.manifold import MDS
    mds = MDS(2,random_state=0,dissimilarity='precomputed')
    X_2d = mds.fit_transform(proximityMatrix)
    colors = ['red','green','blue','black','pink','orange','red','green','blue','black','pink','orange','red','green','blue','black','pink','orange','red','green','blue','black','pink','orange']
    uniqueLables = np.unique(actualLables)
    for i,lab in enumerate(uniqueLables):
        subset = X_2d[actualLables == lab]
    
        x = [row[0] for row in subset]
        y = [row[1] for row in subset]
        # plt.scatter(x,y,c=colors[i],label=featurelables[i])
        plt.scatter(x,y,c=colors[i],label=str(lab))
    plt.legend()
    # plt.show()
    plt.savefig(filename,bbox_inches='tight')


def plotMultiModel(model_names,param_name,param_values,training_results,testing_results,oob_results,filename):
    fig = plt.figure(figsize=(12, 7.5))
    ax=plt.axes()
    ax.set_xlabel(param_name)
    ax.set_ylabel("score")

    # The x axis is the samee for all the plots and is the 'moving' paramter we are experimenting on
    X = param_values

    # Plot the training results 
    for itrain,results in enumerate(training_results):
        theLabel = model_names[itrain]+"_train"
        Y = results
        # for error instead : flip it
        # Y = 1 - Y
        ax.plot(X,Y,label=theLabel)

    # Plot the test results 
    for itest,results in enumerate(testing_results):
        theLabel = model_names[itest]+"_test"
        Y = results
        # for error instead : flip it
        # Y = 1 - Y
        ax.plot(X,Y,label=theLabel,linestyle="dashed")

    # Plot the OOB results (if any)
    for itest,results in enumerate(oob_results):
        theLabel = model_names[itest]+"oob"
        Y = results
        # for error instead : flip it
        # Y = 1 - Y

        # if oob missing vlaues will be set to -1, so chceck for that and avoid plotting lines at -1!!!
        if Y[0]>-1:
            ax.plot(X,Y,label=theLabel,linestyle="dotted")

    plt.legend(loc="lower right")


    # plt.show()
    plt.savefig(filename,bbox_inches='tight')
