#!/usr/bin/env python
# example.py

from Helper import *
import sys
import math

if __name__=="__main__":

    # Settings ------------------------------------------------------------------------------------
    
    trainingFile      = 'Data/BdtBeamParticleId_Training_Beam_Cosmics_5GeV_Concatenated.txt'
    trainingMomentum  = 5
    testingFiles      = []
    testingFiles      = [{'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-7GeV_Concatenated.txt', 'Momentum' : -7 },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-5GeV_Concatenated.txt', 'Momentum' : -5  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_-1GeV_Concatenated.txt', 'Momentum' : -1  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_1GeV_Concatenated.txt', 'Momentum' : 1  },
                         {'File' : 'Data/BdtBeamParticleId_Training_Beam_Cosmics_7GeV_Concatenated.txt', 'Momentum' : 7  }]
    bdtName           = 'BdtBeamParticleID'
    treeDepth         = int(sys.argv[1]) #3
    nTrees            = int(sys.argv[2]) #100
    
    serializeToPkl    = True
    serializeToXml    = True
    loadFromPkl       = False
    xmlFileName       = 'BdtBeamParticleID_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.xml'
    pklFileName       = 'BdtBeamParticleID_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.pkl'
    
    #----------------------------------------------------------------------------------------------

    if loadFromPkl:
        OverwriteStdout('Loading model from file ' + pklFileName + '\n')
        bdtModel = LoadFromPkl(pklFileName)
    
    else:
        # Load the training data
        OverwriteStdout('Loading training set data from file ' + trainingFile + '\n')
        trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')

        # Test Data in useable format
        #X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)
        X_org, Y_org = SplitDetailedTrainingSet(trainSet, nFeatures)
        #DrawVariables(X_org, Y_org)
        #Correlation(X_org, Y_org)

        #sys.exit()
        
        # Train the BDT
        X, Y = Randomize(X_org, Y_org)
        X_train, Y_train, X_test, Y_test = Sample(X, Y, 0.5)
        
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
        plot_range = (-1,1)
        nBins = 100

        # Find optimal cut based on significance
        train_results = bdtModel.decision_function(X_train)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(2, 3, 1)
        ax.set_title(str(trainingMomentum) + " Gev Beam Cosmic, Remainder Training Sample")

        sigEff = 0
        bkgRej = 0
        sigEntries = []
        bkgEntries = []

        for i, n, g in zip(signal_definition, class_names, plot_colors):
            entries, bins, patches = ax.hist(train_results[Y_train == i],
                                             bins=nBins,
                                             range=plot_range,
                                             facecolor=g,
                                             label='Class %s' % n,
                                             alpha=.5,
                                             edgecolor='k')
            if i == 1:
                sigEntries = entries
            elif i == 0:
                bkgEntries = entries

        nSigEntries = sum(sigEntries)
        nBkgEntries = sum(bkgEntries)
        optimalSignif = 0
        optimalSigEff = 0
        optimalBkgRej = 0
        optimalBinCut = 0
        optimalScoreCut = 0

        for binCut in range(0, nBins):
            nSigPassing = sum(sigEntries[binCut:])
            nBkgPassing = sum(bkgEntries[binCut:])

            signif = 0
            if (nSigPassing + nBkgPassing > 0):
                signif = nSigPassing / math.sqrt(nSigPassing + nBkgPassing)

            if (signif > optimalSignif):
                nBkgFailing = sum(bkgEntries[:binCut])
                sigEff = 100 * nSigPassing / nSigEntries
                bkgRej = 100 * nBkgFailing / nBkgEntries
                optimalSignif = signif
                optimalBinCut = binCut
                optimalScoreCut = bins[optimalBinCut]
                optimalSigEff = sigEff
                optimalBkgRej = bkgRej

        print('Optimal signif : ' + str(optimalSignif))
        print('Optimal sigEff : ' + str(optimalSigEff))
        print('Optimal bkgRej : ' + str(optimalBkgRej))
        print('Optimal binCut : ' + str(optimalBinCut))
        print('Optimal scoreCut : ' + str(optimalScoreCut))

        # Testing BDT Using Remainder of Training Sample
        test_results = bdtModel.decision_function(X_test)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(2, 3, 1)
        ax.set_title(str(trainingMomentum) + " Gev Beam Cosmic, Remainder Training Sample")

        sigEff = 0
        bkgRej = 0

        for i, n, g in zip(signal_definition, class_names, plot_colors):
            entries, bins, patches = ax.hist(test_results[Y_test == i],
                                             bins=nBins,
                                             range=plot_range,
                                             facecolor=g,
                                             label='Class %s' % n,
                                             alpha=.5,
                                             edgecolor='k')
            if i == 1:
                nEntries = sum(entries)
                nEntriesPassing = sum(entries[optimalBinCut:])
                sigEff = nEntriesPassing/nEntries
            elif i == 0:
                nEntries = sum(entries)
                nEntriesFailing = sum(entries[:optimalBinCut])
                bkgRej = nEntriesFailing/nEntries

        plt.text(0.75, 0.75, "Sig Eff {:.4%}, \nBkg Rej {:.4%}, \nScore Cut {:.2}".format(sigEff,bkgRej,optimalScoreCut),
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
            #X_test_data, Y_test_data = SplitTrainingSet(testSet, nFeaturesTest)
            X_test_data, Y_test_data = SplitDetailedTrainingSet(testSet, nFeaturesTest)
            test_results = bdtModel.decision_function(X_test_data)

            sigEff = 0
            bkgRej = 0

            for i, n, g in zip(signal_definition, class_names, plot_colors):
                entries, bins, patches = ax.hist(test_results[Y_test_data == i],
                                                  bins=nBins,
                                                  range=plot_range,
                                                  facecolor=g,
                                                  label='Class %s' % n,
                                                  alpha=.5,
                                                  edgecolor='k')
                if i == 1:
                    nEntries = sum(entries)
                    nEntriesPassing = sum(entries[optimalBinCut:])
                    sigEff = nEntriesPassing/nEntries
                elif i == 0:
                    nEntries = sum(entries)
                    nEntriesFailing = sum(entries[:optimalBinCut])
                    bkgRej = nEntriesFailing/nEntries

            plt.text(0.75, 0.75, "Sig Eff {:.4%}, \nBkg Rej {:.4%}, \nScore Cut {:.2}".format(sigEff,bkgRej,optimalScoreCut),
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
        #plt.show()

