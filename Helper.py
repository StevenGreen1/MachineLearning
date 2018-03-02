#!/usr/bin/env python
# PandoraMVA.py

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import preprocessing
from datetime import datetime

import numpy as np
import sys
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns

def LoadData(trainingFileName, delimiter=','):
    # Use the first example to get the number of columns
    with open(trainingFileName) as file:
        ncols = len(file.readline().split(delimiter))
        
    # First column is a datestamp, so skip it
    trainingSet = np.genfromtxt(trainingFileName, delimiter=delimiter, usecols=range(1,ncols), 
                                dtype=None)
                                
    nExamples = trainingSet.size
    nFeatures = ncols - 2 # last column is the response
    return np.array(trainingSet), nFeatures, nExamples

#--------------------------------------------------------------------------------------------------

def SplitTrainingSet(trainingSet, nFeatures):
    X=[] # features sets
    Y=[] # responses

    for example in trainingSet:
        Y.append(int(example[nFeatures])) # type of Y should be bool or int
        features = []
        for i in range(0, nFeatures):
            features.append(float(example[i])) # features in this model must be Python float
            
        X.append(features)

    return np.array(X).astype(np.float64), np.array(Y).astype(np.int)
    
#--------------------------------------------------------------------------------------------------

def DrawVariables(X, Y):
    plot_colors = ['b', 'r']
    plot_step = 1.0
    class_names = ['Beam Particle', 'Cosmic Rays']
    signal_definition = [1, 0]

    num_rows, num_cols = X.shape
    for feature in range(0, num_cols):
        plot_range = (X[:,feature].min(), X[:,feature].max()) 

        for i, n, g in zip(signal_definition, class_names, plot_colors):
            entries, bins, patches = plt.hist(X[:,feature][Y == i],
                                              bins=50,
                                              range=plot_range,
                                              facecolor=g,
                                              label='Class %s' % n,
                                              alpha=.5,
                                              edgecolor='k')
        plt.yscale('log')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 * 1.1))
        plt.legend(loc='upper right')
        plt.ylabel('Entries')
        plt.xlabel('Variable')

        plt.tight_layout()
        plotName = 'Feature_' + str(feature) + '.pdf'
        plt.savefig(plotName)
        plt.show()
        plt.close()

#--------------------------------------------------------------------------------------------------

def Correlation(X, Y):
    signal = []
    background = []

    for idx, x in enumerate(X):
        if Y[idx] == 1:
            signal.append(x)
        elif Y[idx] == 0:
            background.append(x)

    sig = pd.DataFrame(data=signal) 
    bkg = pd.DataFrame(data=background) 

    # Compute the correlation matrix
    corrSig = sig.corr()
    corrBkg = bkg.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corrSig, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    jet = plt.get_cmap('jet')

    # Signal Plot
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corrSig, mask=mask, cmap=jet, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plotName = 'CorrelationMatrixSignal.pdf'
    plt.savefig(plotName)
    plt.show()
    plt.close()

    # Background Plot
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corrBkg, mask=mask, cmap=jet, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plotName = 'CorrelationMatrixBackground.pdf'
    plt.savefig(plotName)
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------------

def Randomize(X, Y, setSameSeed=False):
    if setSameSeed:
        np.random.seed(0)

    order = np.random.permutation(Y.size)
    return X[order], Y[order]

#--------------------------------------------------------------------------------------------------

def Sample(X, Y, testFraction=0.1):
    trainSize = int((1.0 - testFraction) * Y.size)
    
    X_train = X[:trainSize]
    Y_train = Y[:trainSize]
    X_test  = X[trainSize:]
    Y_test  = Y[trainSize:]
    
    return X_train, Y_train, X_test, Y_test

#--------------------------------------------------------------------------------------------------

def TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=3, max_depthValue=2, learning_rateValue=1.0, 
                           algorithmValue='SAMME', random_stateValue=None):
    # Load the BDT object
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depthValue), 
                             n_estimators=n_estimatorsValue, learning_rate=learning_rateValue, 
                             algorithm=algorithmValue, random_state=random_stateValue) 
    
    # Train the model   
    startTime = time.time() 
    bdt.fit(X_train, Y_train)
    endTime = time.time()

    return bdt, endTime - startTime

#--------------------------------------------------------------------------------------------------

def ValidateModel(model, X_test, Y_test):               
    return model.score(X_test, Y_test)
    
#--------------------------------------------------------------------------------------------------

def OpenXmlTag(modelFile, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>\n')
    return indentation + 4 

#--------------------------------------------------------------------------------------------------

def CloseXmlTag(modelFile, tag, indentation):
    indentation = max(indentation - 4, 0)
    modelFile.write((' ' * indentation) + '</' + tag + '>\n')
    return indentation

#--------------------------------------------------------------------------------------------------

def WriteXmlFeatureVector(modelFile, featureVector, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')

    firstTime=True
    for feature in featureVector:
        if firstTime:
            modelFile.write(str(feature))
            firstTime=False
        else:
            modelFile.write(' ' + str(feature))
            
    modelFile.write('</' + tag + '>\n')
    
#--------------------------------------------------------------------------------------------------

def WriteXmlFeature(modelFile, feature, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')
    modelFile.write(str(feature))     
    modelFile.write('</' + tag + '>\n')

#--------------------------------------------------------------------------------------------------

def WriteXmlFile(filePath, adaBoostClassifer):
    datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    with open(filePath, "a") as modelFile:
        indentation = 0
        indentation = OpenXmlTag(modelFile,    'AdaBoostDecisionTree', indentation)
        WriteXmlFeature(modelFile, 'TestingBDT', 'Name', indentation)
        WriteXmlFeature(modelFile, datetimeString, 'Timestamp', indentation)
        
        for idx, estimator in enumerate(adaBoostClassifer.estimators_):
            boostWeight = adaBoostClassifer.estimator_weights_[idx]
            WriteDecisionTree(estimator, modelFile, indentation, idx, boostWeight)

        CloseXmlTag(modelFile,                 'AdaBoostDecisionTree', indentation)

#--------------------------------------------------------------------------------------------------

def Recurse(node, parentnode, depth, position, indentation, decisionTree, modelFile):
    indentation = OpenXmlTag(modelFile, 'Node', indentation)
    WriteXmlFeature(modelFile, node, 'NodeID', indentation)
    WriteXmlFeature(modelFile, parentnode, 'ParentID', indentation)

    if decisionTree.feature[node] != _tree.TREE_UNDEFINED:
        name = decisionTree.feature[node] #(int)(node) #feature_name[node]
        threshold = decisionTree.threshold[node]
        WriteXmlFeature(modelFile, name, 'VariableID', indentation)
        WriteXmlFeature(modelFile, threshold, 'Threshold', indentation)
        WriteXmlFeature(modelFile, decisionTree.children_left[node], 'LeftDaughterID', indentation)
        WriteXmlFeature(modelFile, decisionTree.children_right[node], 'RightDaughterID', indentation)
        indentation = CloseXmlTag(modelFile, 'Node', indentation)
        indentation = indentation + 4
        Recurse(decisionTree.children_left[node], node, depth + 1, 'Left', indentation, decisionTree, modelFile)
        Recurse(decisionTree.children_right[node], node, depth + 1, 'Right', indentation, decisionTree, modelFile)
        indentation = indentation - 4
    else:
        result = decisionTree.value[node]
        if (result.tolist()[0][1] > result.tolist()[0][0]):
            WriteXmlFeature(modelFile, 'true', 'Outcome', indentation)
        else:
            WriteXmlFeature(modelFile, 'false', 'Outcome', indentation)
        
        indentation = CloseXmlTag(modelFile, 'Node', indentation)

#--------------------------------------------------------------------------------------------------

def WriteDecisionTree(estimator, modelFile, indentation, treeIdx, boostWeight):
    decisionTree = estimator.tree_
    indentation = OpenXmlTag(modelFile, 'DecisionTree', indentation)

    WriteXmlFeature(modelFile, treeIdx, 'TreeIndex', indentation)
    WriteXmlFeature(modelFile, boostWeight, 'BoostWeight', indentation)
    Recurse(0, -1, 1, 'Start', indentation, decisionTree, modelFile)

    indentation = CloseXmlTag(modelFile, 'DecisionTree', indentation)

#--------------------------------------------------------------------------------------------------

def OverwriteStdout(text):
    sys.stdout.write('\x1b[2K\r' + text)
    sys.stdout.flush()

#--------------------------------------------------------------------------------------------------
    
def SerializeToPkl(fileName, model):
    with open(fileName, 'wb') as f:
        pickle.dump(model, f)
    
#--------------------------------------------------------------------------------------------------
    
def LoadFromPkl(fileName):
    with open(fileName, 'rb') as f:
        model = pickle.load(f) 
        
        return model
