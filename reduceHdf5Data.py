#pathToData = './COMPASOutput.h5'


import sys

import h5py  as h5   #for handling data format
import numpy as np  #for array handling
import os           #For checking existence data
import WriteH5File



def reduceH5file(pathToData, pathToNewData):

    # read data
    Data  = h5.File(pathToData)
    print("The main files I have at my disposal are:\n",list(Data.keys()))


    # print("The main files I have at my disposal are:\n",list(Data['formationChannels'].keys()))

	# Which Files do I want?
	# options: ['BSE_RLOF', 'XRayBinaries', 'commonEnvelopes', 'cppSource', 'BSE_Double_Compact_Objects', 'formationChannels', 'pulsarEvolution', 'runtimes', 'supernovae', 'BSE_System_Parameters'])
    filesOfInterest   = {1:'BSE_Double_Compact_Objects',2:'BSE_System_Parameters'}

	# #Give a list of columns you want, if you want all, say ['All']
    columnsOfInterest = {1:['All'], 2:['All']}
    """
    columnsOfInterest =	   {1:['COCoreMassDCOFormation1', 'COCoreMassDCOFormation2',  'ECSNPrimary', 'ECSNSecondary', \
							 'HeCoreMassDCOFormation1', 'HeCoreMassDCOFormation2', 'ID', 'M1', 'M1ZAMS', 'M2', 'M2ZAMS', 'Metallicity1', 'Metallicity2', 'PISNPrimary', 'PISNSecondary', 'PPISNPrimary', 'PPISNSecondary', \
							 'PrimaryMTCase', 'RL1to2PostCEE', 'RL1to2PreCEE', 'RL2to1PostCEE', 'RL2to1PreCEE', 'RLOFSecondaryAfterCEE', 'SecondaryMTCase', 'SemiMajorAxisPostCEE', 'SemiMajorAxisPreCEE', 'USSNPrimary', 'USSNSecondary',\
							  'coreMassDCOFormation1', 'coreMassDCOFormation2', 'doubleCommonEnvelopeFlag', 'drawnKick1', 'drawnKick2', 'eccentricityDCOFormation', 'eccentricityInitial', 'eccentricityPrior2ndSN',\
							    'kickDirectionPower', 'lambda1CE', 'lambda2CE', 'mergesInHubbleTimeFlag', 'optimisticCEFlag', 'phiSupernova1', 'phiSupernova2',   'recycledPrimary', 'recycledSecondary', 'relativeVelocity2ndSN', 'samplingPhase', 'seed', \
							    'separationDCOFormation', 'separationInitial', 'separationPrior2ndSN', 'sigmaKickBH', 'sigmaKickNS', 'stellarType1', 'stellarType2', 'tc', 'tform', 'thetaSupernova1', 'thetaSupernova2', 'totalMassDCOFormation1', 'totalMassDCOFormation2', 'weight'],\
							2:['ID', 'Metallicity1', 'Metallicity2', 'SEED', 'disbound', 'eccentricity',  'mass1', 'mass2', 'meanAnomaly1', 'meanAnomaly2', 'omega1', 'omega2', 'phi1', 'phi2', 'rk1', 'rk2', 'samplingPhase', 'separation', 'stellar_merger', 'theta1', 'theta2', 'weight'],\
							3:['MassCOCoreSN', 'MassCoreSN', 'MassStarCompanion', 'MassStarSN',  'Survived','drawnKickVelocity', 'eccentricityAfter', 'eccentricityBefore', 'experiencedRLOF', 'fallback', 'flagECSN', 'flagHpoorSN', 'flagHrichSN', 'flagPISN', 'flagPPISN', 'flagRLOFontoaNS', 'flagSN', 'flagUSSN', 'kickVelocity', \
								  'phi', 'previousStellarTypeCompanion', 'previousStellarTypeSN', 'psi', 'randomSeed', 'runawayFlag', 'separationAfter', 'separationBefore', 'systemicVelocity', 'theta', 'time', 'uK', 'vRel', 'whichStar'],\
							4:['All'] \
							}
    """

	# #example of the seeds dictionary the actual one will be defined later
	# seedsOfInterest   = {1:None,\
	#                      2:None,\
	#                      3:None}
    
    seedsDCO = Data['BSE_Double_Compact_Objects']['SEED'][()][:1000000]
    seedsSystems = Data['BSE_System_Parameters']['SEED'][()][:1000000]
    #seedsSN = Data['supernovae']['randomSeed'][()]
    #seedsFC = Data['formationChannels']['m_randomSeed'][()]


    seedsOfInterest   = {1:seedsDCO,\
                          2:seedsSystems}

    WriteH5File.reduceH5(pathToOld = pathToData, pathToNew = pathToNewData,\
                     dictFiles=filesOfInterest, dictColumns=columnsOfInterest, dictSeeds=seedsOfInterest)



if __name__ == "__main__":
    pathToData = (sys.argv[1])
    pathToNewData = (sys.argv[2])
    
#    print('test')
    reduceH5file(pathToData, pathToNewData)




