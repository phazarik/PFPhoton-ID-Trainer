#############################################################################
#                               PLOTMAKER                                   #
#                                                                           #
# This code reads into CSV Files, determines what is signal and background, #
# and makes plots of the variables.                                         #
# it should be run as follows :                                             #
#                 python PFPhoton-ID-Plotmaker <foldername>                 #
#                                                                           #
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
import os

sample = sys.argv[1]
os.system("")
os.system("mkdir -p plots/" + sample + "/")
txt = open(f'plots/' + sample + f'/Info.txt', "w+")

#Do you want barrel or endcap?
isBarrel = False
#Do you want to debug?
isDebug = False #True -> nrows=1000 #False -> all available photons
#Do you want the plots to be normalised?
isNorm = False   #True -> Normalised plots

###################################################################################################################################################################

#READING DATA FILES :
print('Reading the input files')
#Columns: "phoPt", "phoEta", "phoPhi", "phoHoverE", "phohadTowOverEmValid", "photrkSumPtHollow", "photrkSumPtSolid", "phoecalRecHit", "phohcalTower", "phosigmaIetaIeta", 'phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5', "isPhotonMatching", "isPionMother", "isPFPhoton" (+ "sample" , "label" added later)

mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file1 = file1.drop(['isPionMother'], axis=1)
    file2 = pd.read_csv('../TrainingSamples/df_GJet_20toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file2 = file2.drop(['isPionMother'], axis=1)
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
    file2 = pd.read_csv('../TrainingSamples/df_GJet_20toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    file2 = file2.drop(['isPionMother'], axis=1)
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
signal2 = file2[file2['isPhotonMatching'] ==1 ]
#signal3 = file3[file3['isPhotonMatching'] ==1 ]

#######################################
# Defining the Background data-frames #
#######################################

print('\nDefining the Background File')

#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] ==0 ] 
background2 = file2[file2['isPhotonMatching'] ==0 ]
#background3 = file3[file3['isPhotonMatching'] ==0 ]
background4 = file4[file4['isPhotonMatching'] ==0 ]

##################################################################
#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
signal2["sample"]=1
#signal3["sample"]=2
#background1["sample"]=0
background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3

#signal1["label"]=1
signal2["label"]=1
#signal3["label"]=1
#background1["label"]=0
background2["label"]=0
#background3["label"]=0
background4["label"]=0

################################################################
#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal2])

if isBarrel == True :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel only
else :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap only

#Manually reducing signals :
#Sig_alldf=Sig_alldf.sample(frac=1).reset_index(drop=True) #randomizing the rows 
#Sig_alldf=Sig_alldf.head(1000000) #Keeps only the first 1 million rows

print('\nSignal Photons :')
print(Sig_alldf.shape)

Bkg_alldf = pd.concat([background2, background4])
if isBarrel == True :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel only
else :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap only

print('\nBackground Photons :')
print(Bkg_alldf.shape)

#########################################################
#                  variable plotting                    #
#########################################################

#List of variables in the dataframes at this point :
# Not used in training : "phoPt", "phoEta", "phoPhi" "isPhotonMatching", "isPFPhoton","sample" , "label", "phohadTowOverEmValid", "photrkSumPtSolid"
# Used in training : "phoHoverE", "photrkSumPtHollow", "phoecalRecHit", "phohcalTower", "phosigmaIetaIeta", 'phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5'

#########################
#Plot 1: phoPt
Pt_bins = np.arange(0, 151, 1)
plt.figure(figsize=(8,6))
if isNorm == True :
    plt.hist(Sig_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=True,log=False)
    plt.hist(Bkg_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=True,log=False)
else :
    plt.hist(Sig_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
    plt.hist(Bkg_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False)
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Photon Pt (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Photon Pt (endcap)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoPt_{sample}.png')
plt.close()

#######################
#Plot 2: phoEta
Eta_bins = np.arange(-4, 4, 0.1)
plt.figure(figsize=(8,6))
if isNorm == True :
    plt.hist(Sig_alldf['phoEta'], bins=Eta_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=True,log=False)
    plt.hist(Bkg_alldf['phoEta'], bins=Eta_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=True,log=False)
else :
    plt.hist(Sig_alldf['phoEta'], bins=Eta_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
    plt.hist(Bkg_alldf['phoEta'], bins=Eta_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False)
plt.legend(loc='best')
plt.xlabel('phoEta',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Photon Eta (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Photon Eta (endcap)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoEta_{sample}.png')
plt.close()

#Plot 3 phoSigmaIEtaIEtaFull5x5
phoSigmaIEtaIEtaFull5x5_bins = np.arange(0, 0.07, 0.0005)
plt.figure(figsize=(8,6))
if isNorm == True :
    plt.hist(Sig_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=True,log=False)
    plt.hist(Bkg_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=True,log=False)
else :
    plt.hist(Sig_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
    plt.hist(Bkg_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False)
plt.legend(loc='best')
plt.xlabel('phoSigmaIEtaIEtaFull5x5',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (endcap)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoSigmaIEtaIEtaFull5x5_{sample}.png')
plt.close()


############################################
#              Reweighing Pt               #
############################################
# We want to keep the same signal file and want to weigh the background accordingly.
# Weights calculation function:
def find_weight(bin_condition) :
    n_sig = len(Sig_alldf.query(bin_condition))
    n_bkg = len(Bkg_alldf.query(bin_condition))
    weight = n_sig/n_bkg #we can reweigth the other way around by flipping this ratio.
    #weight_list = [weight]*len(Bkg_alldf.query(bin_condition))
    return weight                            

# The goal is to plot the variables in certain bins such that the number of entries is
# roughly the same in each bin for both signal and background.
# Pt bins : 0-30, 30-70 and 70+ (GeV). These are defined as follows.
bin1 = '(phoPt < 30)'
bin2 = '(30 < phoPt) & (phoPt < 70)'
bin3 = '(30 < phoPt)'

# Calculating the weights :
w1 = find_weight(bin1)
w2 = find_weight(bin2)
w3 = find_weight(bin3)
#Add these weights to the background file :
Bkg_alldf['weights'] =1
Bkg_alldf.loc[ (Bkg_alldf['phoPt'] < 30), 'weights' ] = w1
Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 30) & (Bkg_alldf['phoPt'] < 70), 'weights' ] = w2
Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 70), 'weights' ] = w3

print('\n########################')
print('pT WEIGHTS :')
print(f'w1 = {w1}')
print(f'w2 = {w2}')
print(f'w3 = {w3}')
print('########################\n')

txt.write('pT WEIGHTS :\n')
txt.write(f'w1 = {w1}\n')
txt.write(f'w2 = {w2}\n')
txt.write(f'w3 = {w3}\n')
###################################################

#Weighed Pt plot:
plt.figure(figsize=(8,6))
Pt_bins = np.arange(0, 151, 1)
plt.hist(Sig_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
plt.hist(Bkg_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False, weights=Bkg_alldf['weights'].tolist())
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Photon Pt (barrel, weighed Pt)',fontsize=20) #barrel only
else :
    plt.title(f'Photon Pt (endcap, weighedPt)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoPt_{sample}_weighedPt.png')
plt.close()

#Weighed phoSigmaIEtaIEtaFull5x5 plot:
plt.figure(figsize=(8,6))

plt.hist(Sig_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
plt.hist(Bkg_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False, weights=Bkg_alldf['weights'].tolist())
plt.legend(loc='best')
plt.xlabel('phoSigmaIEtaIEtaFull5x5',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (barrel, weighed Pt)',fontsize=20) #barrel only
else :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (endcap, weighedPt)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoSigmaIEtaIEtaFull5x5_{sample}_weighedPt.png')
plt.close()


############################################
#      Reweighing Pt and Eta together       #
############################################
# The goal is to plot the variables in certain bins such that the number of entries is
# roughly the same in each bin for both signal and background.
# Pt bins : 0-30, 30-70 and 70+ (GeV)
# Eta bins : for barrel photons : 0-0.7, 0.7+ ; for endcap photons : 1.566-2.35, 2.35 +
# Total numbers of bins : 3x2 = 6. These are defined as follows.

if isBarrel == True :
    bin_11 = '(phoPt < 30) & (abs(phoEta) < 0.7)'
    bin_12 = '(phoPt < 30) & (0.7 < abs(phoEta))'
    bin_21 = '(30 < phoPt) & (phoPt < 70) & (abs(phoEta) < 0.7)'
    bin_22 = '(30 < phoPt) & (phoPt < 70) & (0.7 < abs(phoEta))'
    bin_31 = '(70 < phoPt) & (abs(phoEta) < 0.7)'
    bin_32 = '(70 < phoPt) & (0.7 < abs(phoEta))'
else :
    bin_11 = '(phoPt < 30) & (abs(phoEta) < 2.35)'
    bin_12 = '(phoPt < 30) & (2.35 < abs(phoEta))'
    bin_21 = '(30 < phoPt) & (phoPt < 70) & (abs(phoEta) < 2.35)'
    bin_22 = '(30 < phoPt) & (phoPt < 70) & (2.35 < abs(phoEta))'
    bin_31 = '(70 < phoPt) & (abs(phoEta) < 2.35)'
    bin_32 = '(70 < phoPt) & (2.35 < abs(phoEta))'

#Pt and Eta reweighing is done when both barrel and endcap photons are considered.
#Therefore, the follwing codes work only when region is set to 'all'

#Calculation of weights:
w_11 = find_weight(bin_11)
w_12 = find_weight(bin_12)
w_21 = find_weight(bin_21)
w_22 = find_weight(bin_22)
w_31 = find_weight(bin_31)
w_32 = find_weight(bin_32)

#Add these weights to the background file :
Bkg_alldf['weights'] =1 #reset weights
if isBarrel == True :
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] < 30) & (Bkg_alldf['phoEta'] < 0.7), 'weights' ] = w_11
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] < 30) & (Bkg_alldf['phoEta'] > 0.7), 'weights' ] = w_12
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 30) & (Bkg_alldf['phoPt'] < 70) & (Bkg_alldf['phoEta'] < 0.7), 'weights' ] = w_21
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 30) & (Bkg_alldf['phoPt'] < 70) & (Bkg_alldf['phoEta'] > 0.7), 'weights' ] = w_22
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 70) & (Bkg_alldf['phoEta'] < 0.7), 'weights' ] = w_31
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 70) & (Bkg_alldf['phoEta'] > 0.7), 'weights' ] = w_32
else :
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] < 30) & (Bkg_alldf['phoEta'] < 2.35), 'weights' ] = w_11
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] < 30) & (Bkg_alldf['phoEta'] > 2.35), 'weights' ] = w_12
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 30) & (Bkg_alldf['phoPt'] < 70) & (Bkg_alldf['phoEta'] < 2.35), 'weights' ] = w_21
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 30) & (Bkg_alldf['phoPt'] < 70) & (Bkg_alldf['phoEta'] > 2.35), 'weights' ] = w_22
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 70) & (Bkg_alldf['phoEta'] < 2.35), 'weights' ] = w_31
    Bkg_alldf.loc[ (Bkg_alldf['phoPt'] > 70) & (Bkg_alldf['phoEta'] > 2.35), 'weights' ] = w_32


print('\n########################')
print('pT and Eta WEIGHTS :')
print(f'w_11 = {w_11}')
print(f'w_12 = {w_12}')
print(f'w_21 = {w_21}')
print(f'w_22 = {w_22}')
print(f'w_31 = {w_31}')
print(f'w_32 = {w_32}')
print('########################\n')

txt.write('\npT and Eta WEIGHTS :\n')
txt.write(f'w_11 = {w_11}\n')
txt.write(f'w_12 = {w_12}\n')
txt.write(f'w_21 = {w_21}\n')
txt.write(f'w_22 = {w_22}\n')
txt.write(f'w_31 = {w_31}\n')
txt.write(f'w_32 = {w_32}\n')
#############################################################################

#Plotting a histogram for different bins with calculated weights :
# weighed Pt plot :
plt.figure(figsize=(8,6))
Pt_bins = np.arange(0, 151, 1)
plt.hist(Sig_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
plt.hist(Bkg_alldf['phoPt'], bins=Pt_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False, weights= Bkg_alldf['weights'].tolist())
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Photon Pt (barrel, weighed Pt and Eta)',fontsize=20) #barrel only
else :
    plt.title(f'Photon Pt (endcap, weighedPtandEta)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoPt_{sample}_weighedPtandEta.png')
plt.close() 
 
#phoSigmaIEtaIEtaFull5x5 plot :
plt.figure(figsize=(8,6))

plt.hist(Sig_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Signal Photons", linewidth=2, color='xkcd:green',density=False,log=False)
plt.hist(Bkg_alldf['phoSigmaIEtaIEtaFull5x5'], bins=phoSigmaIEtaIEtaFull5x5_bins, histtype='step', label="Background Photons", linewidth=2, color='xkcd:denim',density=False,log=False, weights= Bkg_alldf['weights'].tolist())
plt.legend(loc='best')
plt.xlabel('phoSigmaIEtaIEtaFull5x5',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (barrel, weighed Pt and Eta)',fontsize=20) #barrel only
else :
    plt.title(f'phoSigmaIEtaIEtaFull5x5 (endcap, weighedPtandEta)',fontsize=20) #endcap only
plt.savefig(f'plots/' + sample + f'/phoSigmaIEtaIEtaFull5x5_{sample}_weighedPtandEta.png')
plt.close()

################################################################################
# Printing information :
n_sig = len(Sig_alldf)
n_bkg = len(Bkg_alldf)
print('\n#########################################')
print(f'Number of Signal Photons = {n_sig}')
print(f'Number of Background Photons = {n_bkg}')
print('#########################################\n')
txt.write(f'\nNumber of Signal Photons = {n_sig}\n')
txt.write(f'Number of Background Photons = {n_bkg}\n')

# Plotting complete
#################################################################################
txt.close()
print(f'\nAll done. Plots are saved in the folder plots/{sample}\n')
