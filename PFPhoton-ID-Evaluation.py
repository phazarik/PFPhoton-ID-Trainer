#############################################################################
#                   TESTING A SEQUENTIAL NEURAL NETWORK                     #
#                                                                           #
# This code reads into testing CSV Files and the trained model.h5 file.     #
# It tests the already trained NN on the CSV files                          #
# it should be run as follows :                                             #
#                python PFPhoton-ID-Evaluation.py <modelname>               #
# NOTE : Use the max and minvalues from the training step for normalisation #
#############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import warnings
warnings.filterwarnings('ignore')
import sys

os.system("")
modelname = sys.argv[1]
os.system("mkdir -p evaluated/" + modelname)
txt = open(f'evaluated/' + modelname+ f'/Info_{modelname}.txt', "w+")

#Do you want barrel or endcap?
isBarrel = True #True -> Barrel, False -> Endcap
#Do you want to debug?
isDebug = False #True -> nrows=1000

###################################################################################################################################################################

#READING DATA FILES :
#Columns: "phoPt", "phoEta", "phoPhi", "phoHoverE", "phohadTowOverEmValid", "photrkSumPtHollow", "photrkSumPtSolid", "phoecalRecHit", "phohcalTower", "phosigmaIetaIeta", 'phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5', "isPhotonMatching", "isPionMother", "isPFPhoton" (+ "sample" , "label" added later)

print('Reading the input files')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file1 = file1.drop(['isPionMother'], axis=1)
    #file2 = pd.read_csv('../TrainingSamples/df_GJet_20toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file2 = file2.drop(['isPionMother'], axis=1)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file3 = file3.drop(['isPionMother'], axis=1)
    file4 = pd.read_csv('../TrainingSamples/df_QCD_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file4 = file4.drop(['isPionMother'], axis=1)
    #file5 = pd.read_csv('../TrainingSamples/df_TauGun_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file5 = file5[file5['isPionMother'] == 1] # photons which are coming for a pion (part of the background)
    #file5 = file5.drop(['isPionMother'], axis=1)
else : #take all the photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols)
    #file1 = file1.drop(['isPionMother'], axis=1)
    #file2 = pd.read_csv('../TrainingSamples/df_GJet_20toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    #file2 = file2.drop(['isPionMother'], axis=1)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    #file3 = file3.drop(['isPionMother'], axis=1)
    file4 = pd.read_csv('../TrainingSamples/df_QCD_20.csv.gzip',compression='gzip', usecols=mycols)
    file4 = file4.drop(['isPionMother'], axis=1)
    #file5 = pd.read_csv('../TrainingSamples/df_TauGun_20.csv.gzip',compression='gzip', usecols=mycols)
    #file5 = file5[file5['isPionMother'] == 1] # photons which are coming for a pion (part of the background)
    #file5 = file5.drop(['isPionMother'], axis=1)

##################################################################################################################################################################
#  Defining the Signal dataframes   #
#####################################

print('\nDefining the Signal File')

#signal1 = file1[file1['isPhotonMatching'] ==1 ]
#signal2 = file2[file2['isPhotonMatching'] ==1 ]
#signal3 = file3[file3['isPhotonMatching'] ==1 ]
signal4 = file4[file4['isPhotonMatching'] ==1 ]

#######################################
# Defining the Background data-frames #
#######################################

print('\nDefining the Background File')

#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] ==0 ] 
#background2 = file2[file2['isPhotonMatching'] ==0 ]
#background3 = file3[file3['isPhotonMatching'] ==0 ]
background4 = file4[file4['isPhotonMatching'] ==0 ]


##################################################################
#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
#signal2["sample"]=1
#signal3["sample"]=2
signal4["sample"]=3
#background1["sample"]=0
#background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3

#signal1["label"]=1
#signal2["label"]=1
#signal3["label"]=1
signal4["label"]=1
#background1["label"]=0
#background2["label"]=0
#background3["label"]=0
background4["label"]=0

################################################################
#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal4])

if isBarrel == True :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel only
else:
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap only

#Manually reducing signals :
#Sig_alldf=Sig_alldf.sample(frac=1).reset_index(drop=True) #randomizing the rows 
#Sig_alldf=Sig_alldf.head(1000000) #Keeps only the first 1 million rows

print('\nSignal Photons :')
print(Sig_alldf.shape)

Bkg_alldf = pd.concat([background4])
if isBarrel == True :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel only
else :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap only

print('\nBackground Photons :')
print(Bkg_alldf.shape)

#########################################################
#removing the unnecessary columns :

print("Removing Unnecessary Columns from Signal and Background Dataframes")
Sigdf = Sig_alldf.drop(['phoPt','phoEta','phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid'], axis=1)
Bkgdf = Bkg_alldf.drop(['phoPt','phoEta','phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid'], axis=1)
print('\nData reading succesful !\nBEGINNING THE TRAINING PROCESS\n')

##########################################################

#Final data frame creaton :    
data = pd.concat([Sigdf, Bkgdf])

print('dataframes are succesfully created.')
print('\nTotal Photons :')
print(data.shape)
print(data.head)

n_sig = len(data.query('label == 1'))
n_bkg = len(data.query('label == 0'))
print(f'Number of Signal Photons = {n_sig}\n')
print(f'Number of Background Photons = {n_bkg}\n')
txt.write(f'Number of Signal Photons = {n_sig}\n')
txt.write(f'Number of Background Photons = {n_bkg}\n')


X_sig, y_sig = Sigdf[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values, Sigdf[['label']].values
X_bkg, y_bkg = Bkgdf[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values, Bkgdf[['label']].values
#We don't need to split it, since we are only evaluating the NN

#########################################################
#                     NORMALISATION                     #
#########################################################
#The following two lists are to be manually put from the training step.
maxValues = [0.14999168, 13607.641, 178.11313, 0.024679672, 0.030114882, 0.0007522316, 188.55429, 192.22289, 1.0, 12.263091, 117.800644]
minValues = [0.0, 0.0, 0.0, 0.0, 0.0, -0.0006832788, 0.0, 0.0, 0.0, 0.1315181, 0.0]

MaxMinusMin = []
entries = 0
while entries<len(maxValues):
    difference = maxValues[entries]-minValues[entries]
    MaxMinusMin.append(difference)
    entries = entries + 1
    
normedX_sig = 2*((X_sig - minValues)/(MaxMinusMin)) -1.0
X_sig = normedX_sig
normedX_bkg = 2*((X_bkg - minValues)/(MaxMinusMin)) -1.0
X_bkg = normedX_bkg
print("The data has been normalised.")

########################################################
#             Loading the neural network               #
########################################################

mymodel = tf.keras.models.load_model('output/'+ modelname + '/' + modelname + '.h5')
mymodel.load_weights('output/'+ modelname + '/' + modelname + '.h5')

def get_roc_details(model,Xbk,ybk,Xsig,ysig):
    bk_proba = model.predict(Xbk)
    sig_proba= model.predict(Xsig)
    proba_tot = np.concatenate((sig_proba,bk_proba),axis=0)
    class_tot = np.concatenate((ysig,ybk),axis=0)
    fpr, tpr, _ = roc_curve(class_tot,proba_tot)
    aucscore = auc(tpr,1-fpr)
    tpr=tpr*100
    fnr=(1-fpr)*100
    return fnr,tpr,aucscore

myfnr, mytpr, myauc = get_roc_details(mymodel,X_sig,y_sig,X_bkg,y_bkg)


########################################################
#                      Plotting                        #
########################################################
plt.figure(figsize=(8,8))

#Plotting the CMSSW point
backgroundpass=len(data.query('(isPFPhoton == 1) & (label == 0)'))
backgroundrej =len(data.query('(isPFPhoton == 0) & (label == 0)'))
signalpass    =len(data.query('(isPFPhoton == 1) & (label == 1)'))
signalrej     =len(data.query('(isPFPhoton == 0) & (label == 1)'))
backgroundrej =( backgroundrej/(backgroundpass+backgroundrej) )*100
signaleff     =( signalpass/(signalpass+signalrej) )*100
plt.plot([signaleff], [backgroundrej], marker='o', color="red", markersize=6, label='CMSSW flag')

#Plotting the ROC:
plt.plot(mytpr,myfnr,color='xkcd:bright blue',label='QCD (Testing AUC = %0.4f)' % myauc)
plt.legend(loc='lower right')

if isBarrel == True :
    plt.title(f'ROC curve (testing, barrel)',fontsize=20)
else :
    plt.title(f'ROC curve (testing, endcap)',fontsize=20)

plt.xlabel('Signal Efficiency',fontsize=20)
plt.ylabel('Background Rejection',fontsize=20)
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('evaluated/' + modelname + '/' + modelname + '.png')

plt.close()
print(f'\nAll done. Evaluated ROC is saved in the folder : evaluated/{modelname}\n')
