#!/usr/bin/env python
# example.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from Helper import *

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__=="__main__":

    trainingFile = 'Data/BdtBeamParticleId_Training_Beam_Cosmics_1GeV_Concatenated.txt'

    # Load the data
    OverwriteStdout('Loading training set data from file ' + trainingFile + '\n')
    trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')

    X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)

    nTreesRange = np.array([1,5,10])
    treeDepthRange = np.array([2,3,4])
    
    param_grid = { "n_estimators" : nTreesRange,
                   "base_estimator__max_depth" : treeDepthRange}

    crossValidation = StratifiedShuffleSplit(n_splits=2, test_size=0.001, random_state=42)

    decisionTreeClassifier = DecisionTreeClassifier()
    adaBoostClassifier = AdaBoostClassifier(base_estimator = decisionTreeClassifier)
    gridSearchCV = GridSearchCV(adaBoostClassifier, param_grid=param_grid, cv=crossValidation, n_jobs=8)
    gridSearchCV.fit(X_org, Y_org)

    print("The best parameters are %s with a score of %0.2f"
      % (gridSearchCV.best_params_, gridSearchCV.best_score_))
    
    scores = gridSearchCV.cv_results_['mean_test_score'].reshape(len(treeDepthRange), len(nTreesRange))

    print(scores)
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Greens)
    plt.xlabel('Number of Trees')
    plt.ylabel('Tree Depth')
    plt.colorbar()
    plt.xticks(np.arange(len(nTreesRange)), nTreesRange, rotation=45)
    plt.yticks(np.arange(len(treeDepthRange)), treeDepthRange)
    plt.title('Validation Accuracy')
    plt.show()

