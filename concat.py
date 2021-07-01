#############################################################################
#                         READING INTO THE CSV FILES                        #
#   This program reads into the CSV files, labels single and background,    #
#   adds them into a single data frame, and produces a single CSV file      #
#############################################################################
# Note : The output data file can be split manually into training and
# testing part using the following command :
# sed -n 1,<line no. till half>p data.csv > train.csv
# sed -n <line no. till half+1>,<lastline no.>p data.csv > test.csv


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


print('Reading the input files')
#Reading the data :
#signal1 = pd.read_csv('../TrainingSamples/GJet_20to40/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
signal2 = pd.read_csv('../TrainingSamples/GJet_20toinf/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
#signal3 = pd.read_csv('../TrainingSamples/GJet_40toinf/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

background1=pd.read_csv('../TrainingSamples/QCD/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
background2=pd.read_csv('../TrainingSamples/TauGun/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

##########################################################
#                    PF PHOTON FLAGS                     #
#                                                        #
# PF Flag Parameters are defined here:                   #
photon_min_Et =10.0;                                     #
photon_HoE = 0.05;                                       #
photon_comb_Iso = 10.0;                                  #
solidConeTrackIsoSlope = 0.3;                            #
solidConeTrackIsoOffset = 10.0;                          #
photon_sigmaIetaIeta_barrel=0.0125;   #for barrel part   #
photon_sigmaIetaIeta_endcap=0.034;    #for endcap        #
#                                                        #
#Defining the flags below in function form.              #
#They give a true value if the Photon is a PF-photon     #
# 1 = True (We want these as int values later)           #
# 0 = False                                              #
def flag1(phoPt) :                                       #
    if phoPt > photon_min_Et :                           #
        return 1
    else :
        return 0

def flag2(phoHoverE) :
    if phoHoverE < photon_HoE :
        return 1
    else :
        return 0

def flag3(photrkSumPtHollow, phoecalRecHit, phohcalTower) :
    if photrkSumPtHollow + phoecalRecHit + phohcalTower < photon_comb_Iso :
        return 1
    else :
        return 0

def flag4(phohadTowOverEmValid, photrkSumPtSolid, phoPt) :
    if phohadTowOverEmValid != 0 and photrkSumPtSolid >solidConeTrackIsoOffset + solidConeTrackIsoSlope*phoPt :
        return 0
    else :
        return 1
    
def flag5(phoEta, phosigmaIetaIeta) :
    if abs(phoEta) < 1.442 : #for barrel photons
        if phosigmaIetaIeta < photon_sigmaIetaIeta_barrel :
            return 1
        else :
            return 0
    elif abs(phoEta) > 1.566 : #for endcap photons
        if phosigmaIetaIeta < photon_sigmaIetaIeta_endcap :
            return 1
        else :
            return 0
    else : #for the photons within the gap               #
        return 0                                         #
#                                                        #
#                                                        #    
##########################################################

#######################################################################################################################
print('\nReading into the Signal File')

Sig_alldf = pd.concat([signal2])
Sig_alldf = Sig_alldf[Sig_alldf['isPhotonMatching'] == 1]#keep the rows that contain gen-matching photons
#Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel
#Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap

print('Adding the PF flags to the signal file')
Sig_alldf['isPFphoton'] = Sig_alldf.apply(lambda row: flag1(row.phoPt)*flag2(row.phoHoverE)*flag3(row.photrkSumPtHollow, row.phoecalRecHit, row.phohcalTower)*flag4(row.phohadTowOverEmValid, row.photrkSumPtSolid, row.phoPt)*flag5(row.phoEta, row.phosigmaIetaIeta), axis=1)
print('PF Flags are added succesfully!\nNow dropping the unnecessary columns.')
#removing the unnecessary columns :
Sigdf = Sig_alldf.drop(['phoPt','phoEta','phoPhi','isPhotonMatching','isPromptFinalState','phohadTowOverEmValid'], axis=1)
print('Unnecessary columns dropped from the signal File.\nThe dataframe has the following structure :')
print(Sigdf.shape)
print(list(Sigdf.columns))
print(Sigdf.head)

#########################################################################################################################
print('\nReading into the Background File')
Bkg_alldf = pd.concat([background1, background2])

#### Adding conditions to the background file :
Bkg_alldf = Bkg_alldf[Bkg_alldf['isPhotonMatching'] == 0] #keep the rows that contain fake photons
#Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel
#Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap

#Adding PF-flags
print('Adding the PF flags to the background file')
Bkg_alldf['isPFphoton'] = Bkg_alldf.apply(lambda row: flag1(row.phoPt)*flag2(row.phoHoverE)*flag3(row.photrkSumPtHollow, row.phoecalRecHit, row.phohcalTower)*flag4(row.phohadTowOverEmValid, row.photrkSumPtSolid, row.phoPt)*flag5(row.phoEta, row.phosigmaIetaIeta), axis=1)
#print(Bkg_alldf.head)
print('PF Flags are added succesfully!\nNow dropping the unnecessary columns.')
#removing the unnecessary columns :
Bkgdf = Bkg_alldf.drop(['phoPt','phoEta','phoPhi','isPhotonMatching','isPromptFinalState','phohadTowOverEmValid'], axis=1)
print('Unnecessary columns dropped from the background File.\nThe dataframe has the following structure :')
print(Bkgdf.shape)
print(Bkgdf.head)

###########################################################################################################################
Sigdf["label"]=1
Bkgdf["label"]=0
data = pd.concat([Sigdf,Bkgdf])
print('dataframes are succesfully created.')
# We need to print this dataframe as one csv file.
data.to_csv('data.csv')
