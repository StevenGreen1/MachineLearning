#!/usr/bin/env python
# example.py

from Helper import *
import sys

if __name__=="__main__":

    # Settings ------------------------------------------------------------------------------------
    
    trainingFile      = 'Data/BdtBeamParticleId_Training_Beam_Cosmics_5GeV_Concatenated.txt'
    trainingMomentum  = 5
    testingFiles      = [{'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-7GeV_Concatenated.txt', 'Momentum' : -7 },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-5GeV_Concatenated.txt', 'Momentum' : -5  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-1GeV_Concatenated.txt', 'Momentum' : -1  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_1GeV_Concatenated.txt', 'Momentum' : 1  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_7GeV_Concatenated.txt', 'Momentum' : 7  }]
    bdtName           = 'BdtBeamParticleID'
    treeDepth         = 3
    nTrees            = 200
    
    serializeToPkl    = False
    serializeToXml    = False
    loadFromPkl       = False
    xmlFileName       = 'BdtBeamParticleID_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.xml'
    pklFileName       = 'BdtBeamParticleID_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.pkl'
    
#    tol       = 0.001
#    shrinking = False 
    
    #----------------------------------------------------------------------------------------------
    
    if loadFromPkl:
        OverwriteStdout('Loading model from file ' + pklFileName + '\n')
        bdtModel = LoadFromPkl(pklFileName)
    
    else:
        # Load the training data
        OverwriteStdout('Loading training set data from file ' + trainingFile + '\n')
        trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')

        # Test Data in useable format
        X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)
        #DrawVariables(X_org, Y_org)
        Correlation(X_org, Y_org)

        sys.exit()
        
        # Train the BDT
        X, Y = Randomize(X_org, Y_org)
        X_train, Y_train, X_test, Y_test = Sample(X, Y, 0.1)
        
        OverwriteStdout('Training AdaBoostClassifer...')
        bdtModel, trainingTime = TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=nTrees, max_depthValue=treeDepth)

        OverwriteStdout(('Trained AdaBoostClassifer with ' + str(nFeatures) + ' features and ' + 
                         str(nExamples) + ' examples (%d seconds, %i TreeDepth, %i nTrees)\n' % (trainingTime, treeDepth, nTrees)))
                  
        # Validate the model 
        modelScore = ValidateModel(bdtModel, X_test, Y_test)
        OverwriteStdout('Model score: %.2f%%\n' % (modelScore * 100))
        
        if serializeToXml:
            OverwriteStdout('Writing model to xml file ' + xmlFileName + '\n')
            datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            WriteXmlFile(xmlFileName, bdtModel)

        if serializeToPkl:
            OverwriteStdout('Writing model to pkl file ' + pklFileName + '\n')
            SerializeToPkl(pklFileName, bdtModel)
            
        # Do other stuff with your trained/loaded model
        # ...

        plot_colors = ['b', 'r']
        plot_step = 1.0
        class_names = ['Beam Particle', 'Cosmic Rays']
        signal_definition = [1, 0]

        test_results = bdtModel.decision_function(X_test)
        plot_range = (test_results.min(), test_results.max())

        # Testing BDT Using Remainder of Training Sample
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(2, 3, 1)
        ax.set_title(str(trainingMomentum) + " Gev Beam Cosmic, Remainder Training Sample")

        test_results = bdtModel.decision_function(X_test)
        sigEff = 0
        bkgRej = 0

        for i, n, g in zip(signal_definition, class_names, plot_colors):
            entries, bins, patches = ax.hist(test_results[Y_test == i],
                                             bins=10,
                                             range=plot_range,
                                             facecolor=g,
                                             label='Class %s' % n,
                                             alpha=.5,
                                             edgecolor='k')
            if i == 1:
                nEntries = sum(entries)
                nEntriesPassing = sum(entries[5:])
                sigEff = nEntriesPassing/nEntries
            elif i == 0:
                nEntries = sum(entries)
                nEntriesFailing = sum(entries[:5])
                bkgRej = nEntriesFailing/nEntries

        plt.text(0.75, 0.75, "Sig Eff {:.4%}, \nBkg Rej {:.4%}".format(sigEff,bkgRej),
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes) 

        plt.yscale('log')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 * 1.1))
        plt.legend(loc='upper right')
        plt.ylabel('Samples')
        plt.xlabel('Score')

        # Testing BDT Using New Samples
        for idx, testFile in enumerate(testingFiles):
            ax = fig.add_subplot(2, 3, idx+2)
            ax.set_title(str(testFile['Momentum']) + " Gev Beam Cosmic, Full Set")

            testSet, nFeaturesTest, nExamplesTest = LoadData(testFile['File'], ',')
            X_test_data, Y_test_data = SplitTrainingSet(testSet, nFeaturesTest)
            test_results = bdtModel.decision_function(X_test_data)

            sigEff = 0
            bkgRej = 0

            for i, n, g in zip(signal_definition, class_names, plot_colors):
                entries, bins, patches = ax.hist(test_results[Y_test_data == i],
                                                  bins=10,
                                                  range=plot_range,
                                                  facecolor=g,
                                                  label='Class %s' % n,
                                                  alpha=.5,
                                                  edgecolor='k')
                if i == 1:
                    nEntries = sum(entries)
                    nEntriesPassing = sum(entries[5:])
                    sigEff = nEntriesPassing/nEntries
                elif i == 0:
                    nEntries = sum(entries)
                    nEntriesFailing = sum(entries[:5])
                    bkgRej = nEntriesFailing/nEntries

            plt.text(0.75, 0.75, "Sig Eff {:.4%}, \nBkg Rej {:.4%}".format(sigEff,bkgRej),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)

            plt.yscale('log')
            x1, x2, y1, y2 = plt.axis()
            plt.axis((x1, x2, y1, y2 * 1.1))
            plt.legend(loc='upper right')
            plt.ylabel('Samples')
            plt.xlabel('Score')

        plt.tight_layout()
        plt.savefig('TrainingBdtBeamParticleID_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.pdf')
        plt.show()

