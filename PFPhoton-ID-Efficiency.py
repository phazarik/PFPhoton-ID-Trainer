#############################################################################
#                   TESTING A SEQUENTIAL NEURAL NETWORK                     #
#                                                                           #
# This code reads into testing CSV Files and the trained model.h5 file.     #
# It makes efficiency plots in different bins                               #
# it should be run as follows :                                             #
#      python PFPhoton-ID-Efficiency.py <trainedmodelname> <NN-cut>         #
# NOTE : Use the max and minvalues from the training step for normalisation #
#############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse as sparse #for numpy.array - pd.dataframe column conversion
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import warnings
warnings.filterwarnings('ignore')
import sys
import math

os.system("")
modelname = sys.argv[1]
nn_cut = sys.argv[2]
os.system("mkdir -p efficiency/" + modelname + '_' + nn_cut)
txt = open(f'efficiency/' + modelname +'_'+ nn_cut + f'/Info_{modelname}.txt', "w+")

#Do you want barrel or endcap?
isBarrel = True #True -> Barrel, False -> Endcap
#Do you want to debug?
isDebug = False #True -> nrows=1000
#Cut on the Neural Network plot :
NN_cut = float(nn_cut)

###################################################################################################################################################################

#READING DATA FILES :
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton

print('Reading the input files')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file5 = pd.read_csv('../TrainingSamples/df_TauGun_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
else : #take all the photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols)
    #file5 = pd.read_csv('../TrainingSamples/df_TauGun_20.csv.gzip',compression='gzip', usecols=mycols)
    
##################################################################################################################################################################
#  Defining the Signal dataframes   #
#####################################

print('Defining the Signal File')

#signal1 = file1[file1['isPhotonMatching'] ==1 ]
signal2 = file2[(file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 1) ]
#signal3 = file3[file3['isPhotonMatching'] ==1 ]
#signal4 = file4[file4['isPhotonMatching'] ==1 ]

#######################################
# Defining the Background data-frames #
#######################################

print('Defining the Background File')

#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] ==0 ] 
#background2 = file2[file2['isPhotonMatching'] ==0 ]
#background3 = file3[file3['isPhotonMatching'] ==0 ]
background4 = file4[(file4['isPhotonMatching'] == 0) | ((file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 0)) ]

##################################################################
#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
signal2["sample"]=1
#signal3["sample"]=2
#signal4["sample"]=3
#background1["sample"]=0
#background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3

#signal1["label"]=1
signal2["label"]=1
#signal3["label"]=1
#signal4["label"]=1
#background1["label"]=0
#background2["label"]=0
#background3["label"]=0
background4["label"]=0

################################################################
#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal2])
#Sig_alldf = Sig_alldf[ Sig_alldf['isHardProcess'] == 1]

if isBarrel == True :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel only
else:
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap only

#Manually reducing signals :
#Sig_alldf=Sig_alldf.sample(frac=1).reset_index(drop=True) #randomizing the rows 
#Sig_alldf=Sig_alldf.head(1000000) #Keeps only the first 1 million rows

print('Signal Photons dataframe created.')

Bkg_alldf = pd.concat([background4])
if isBarrel == True :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel only
else :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap only

print("Background dataframe created.")

##########################################################
n_sig = len(Sig_alldf)
n_bkg = len(Bkg_alldf)
print('\n#########################################')
print(f'Number of Signal Photons = {n_sig}')
print(f'Number of Background Photons = {n_bkg}')
print('#########################################\n')
txt.write(f'Number of Signal Photons = {n_sig}\n')
txt.write(f'Number of Background Photons = {n_bkg}\n')

X_sig, y_sig = Sig_alldf[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values, Sig_alldf[['label']].values
X_bkg, y_bkg = Bkg_alldf[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values, Bkg_alldf[['label']].values
#We don't need to split it, since we are only evaluating the NN

#########################################################
#                     NORMALISATION                     #
#########################################################
#The following two lists are to be manually put from the training step.

if modelname == 'barrel' :
    #Barrel:
    maxValues = [0.14999595, 9716.979, 1453.3214, 0.025382034, 0.030323075, 0.0006741011, 1469.4491, 1564.4443, 1.0, 40.157692, 1306.034]
    minValues = [0.0, 0.0, 0.0, 0.0, 0.0, -0.00051127985, 0.0, 0.0, 0.0, 0.13221417, 0.0]
elif modelname == 'endcap' :
    #Endcap:
    maxValues = [0.14999561, 45903.42, 1078.8676, 0.08227496, 0.08187144, 0.0027291286, 1080.1742, 1411.9785, 1.0, 33.16064, 777.81445]
    minValues = [0.0, 0.0, 0.0, 0.0, 0.0, -0.0027415548, 0.0, 0.0, 0.0, 0.13407391, 0.067593366]
else :
    print('Please load the correct model. Errors will show up.')

print('### Normalisation Parameters : ###')
print('maxValues :')
print(str(maxValues))
print('minValues :')
print(str(minValues))
print('##################################')

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
print("\nmodel loaded\n")

#Caclculating tpr, fnr and adding the NN score as a separate column to the dataframe:
sig_proba = mymodel.predict(X_sig)
bkg_proba = mymodel.predict(X_bkg)
proba_tot = np.concatenate((sig_proba,bkg_proba),axis=0)
class_tot = np.concatenate((y_sig,y_bkg),axis=0)
fpr, tpr, _ = roc_curve(class_tot,proba_tot)
aucscore = auc(tpr,1-fpr)
tpr=tpr*100
fnr=(1-fpr)*100

print('\nAdding the NN Score to the Sig df')
NNscore_sig = pd.DataFrame(sig_proba)
NNscore_sig.columns = ['NNscore']
Sig_alldf.reset_index(drop = True, inplace= True)
Sig_alldf = pd.merge(Sig_alldf, NNscore_sig, left_index =True, right_index = True)

print('Adding the NN Score to the Bkg df')
NNscore_bkg = pd.DataFrame(bkg_proba)
NNscore_bkg.columns = ['NNscore']
Bkg_alldf.reset_index(drop = True, inplace = True)
Bkg_alldf = pd.merge(Bkg_alldf, NNscore_bkg, left_index =True, right_index = True)
#print(Bkg_alldf.shape)
#print(Bkg_alldf.head)

print('Done! Creating the final dataframe.')
data = pd.concat([Sig_alldf, Bkg_alldf], axis=0)
data = data.drop(['phoPhi', 'phoHoverE', 'phohadTowOverEmValid', 'photrkSumPtHollow', 'photrkSumPtSolid', 'phoecalRecHit', 'phohcalTower', 'phosigmaIetaIeta', 'phoSigmaIEtaIEtaFull5x5', 'phoSigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso', 'phoHcalPFClusterIso', 'phohasPixelSeed', 'phoR9Full5x5', 'isPhotonMatching', 'isPionMother', 'isPromptFinalState', 'isHardProcess'], axis=1)
data.reset_index(drop = True, inplace = True)
print(data.head)

########################################################
#                     Making bins                      #
########################################################

print('\nBinning for efficiency plot.')

Pt_bins=[5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
print(f'Number of Pt bins = {len(Pt_bins)-1}')
if isBarrel == True :
    Eta_bins=[-1.442, -1.08, -0.72, -0.36, 0.0, 0.36, 0.72, 1.08, 1.442]
    print(f'Number of Eta bins = {len(Eta_bins)-1}')
else :
    Eta_bins1=[-3, -2.75, -2.36, -2, -1.566 ]
    Eta_bins2=[1.566, 2, 2.36, 2.75, 3]
    print(f'Number of Eta bins = {len(Eta_bins1)-1 + len(Eta_bins2)-1 }')
data['Pt_bin'] =  pd.cut(data['phoPt'], bins=Pt_bins, labels=list(range(len(Pt_bins)-1)))

if isBarrel == True :
    data['Eta_bin'] = pd.cut(data['phoEta'], bins=Eta_bins, labels=list(range(len(Eta_bins)-1)))
else :
    data['Eta_bin1'] = pd.cut(data['phoEta'], bins=Eta_bins1, labels=list(range(len(Eta_bins1)-1)))
    data['Eta_bin2'] = pd.cut(data['phoEta'], bins=Eta_bins2, labels=list(range(len(Eta_bins2)-1)))
        
print('Data binning success.\n')


txt.write('Pt bins :\n')
txt.write(str(Pt_bins))
txt.write('\nEta bins:\n')
if isBarrel == True :
    txt.write(str(Eta_bins))
else :
    txt.write(str(Eta_bins1))
    txt.write('\n')
    txt.write(str(Eta_bins2))

#########################################################
#Calculation of different efficiencies:

# I can count how many photons are there with label == 1 (signal), Pt_bin == 1  and NNscore > 90 (and so on)
# The effciiency is given by the fraction of photons that pass this cut.

def efficiency(label_, bintype_, binno_, nn_) :
    numerator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_}) & (NNscore > {nn_})'))
    denominator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        print('Some bins have 0 entries, setting eff to zero')
        return 0, 0
    else :
        eff = (numerator)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err
    
#calculatin of global efficiencies at NN_cut (irrespective of bins) :
num_sig = len(data.query(f'(label == 1) &(NNscore > {NN_cut})'))
den_sig = len(data.query(f'(label == 1)'))
num_bkg = len(data.query(f'(label == 0) & (NNscore > {NN_cut})'))
den_bkg = len(data.query(f'(label == 0)'))
if(den_sig > 0):
    sig_eff_global = (num_sig)/den_sig
if(den_bkg > 0):
    bkg_eff_global = (num_bkg)/den_bkg

###############################################################
sig_eff_list = []
sig_eff_list_err =[]
bkg_eff_list = []
bkg_eff_list_err = []
for iterator in range(len(Pt_bins)-1):
    sig_eff, sig_eff_err = efficiency(1, 'Pt_bin', iterator, NN_cut)
    bkg_eff, bkg_eff_err = efficiency(0, 'Pt_bin', iterator, NN_cut)
    sig_eff_list.append(sig_eff)
    bkg_eff_list.append(bkg_eff)
    sig_eff_list_err.append(sig_eff_err)
    bkg_eff_list_err.append(bkg_eff_err)

print('\nThe efficiencies in different Pt bins are as follows.')
print(sig_eff_list)
print(bkg_eff_list)

txt.write('\nsignal_efficiencies :\n')
txt.write(str(sig_eff_list))
txt.write('\nBackground_efficiencies :\n')
txt.write(str(bkg_eff_list))

if isBarrel == True:
    #Eta bins:
    sig_eff_eta = []
    sig_eff_eta_err =[]
    bkg_eff_eta = []
    bkg_eff_eta_err = []
    for iterator in range(len(Eta_bins)-1):
        sig_eff0, sig_eff_err0 = efficiency(1, 'Eta_bin', iterator, NN_cut)
        bkg_eff0, bkg_eff_err0 = efficiency(0, 'Eta_bin', iterator, NN_cut)
        sig_eff_eta.append(sig_eff0)
        bkg_eff_eta.append(bkg_eff0)
        sig_eff_eta_err.append(sig_eff_err0)
        bkg_eff_eta_err.append(bkg_eff_err0)
else :
    sig_eff_eta1 = []
    sig_eff_eta_err1 =[]
    bkg_eff_eta1 = []
    bkg_eff_eta_err1 = []
    for iterator in range(len(Eta_bins1)-1):
        sig_eff1, sig_eff_err1 = efficiency(1, 'Eta_bin1', iterator, NN_cut)
        bkg_eff1, bkg_eff_err1 = efficiency(0, 'Eta_bin1', iterator, NN_cut)
        sig_eff_eta1.append(sig_eff1)
        bkg_eff_eta1.append(bkg_eff1)
        sig_eff_eta_err1.append(sig_eff_err1)
        bkg_eff_eta_err1.append(bkg_eff_err1)

    sig_eff_eta2 = []
    sig_eff_eta_err2 =[]
    bkg_eff_eta2 = []
    bkg_eff_eta_err2 = []
    for iterator in range(len(Eta_bins2)-1):
        sig_eff2, sig_eff_err2 = efficiency(1, 'Eta_bin2', iterator, NN_cut)
        bkg_eff2, bkg_eff_err2 = efficiency(0, 'Eta_bin2', iterator, NN_cut)
        sig_eff_eta2.append(sig_eff2)
        bkg_eff_eta2.append(bkg_eff2)
        sig_eff_eta_err2.append(sig_eff_err2)
        bkg_eff_eta_err2.append(bkg_eff_err2)
        

########################################################
#                      Plotting                        #
########################################################

#Pt_bins=[5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200] #12 numbers -> 11 bins
Pt_bins_err = np.diff(Pt_bins)
Pt_bins_plot = []
i = 0
while i<len(Pt_bins_err):
    x_point = Pt_bins[i] + (Pt_bins_err[i]/2)
    Pt_bins_plot.append(x_point)
    i = i+1

plt.figure(figsize=(8,8))
plt.errorbar(Pt_bins_plot, sig_eff_list, xerr = Pt_bins_err/2, yerr=sig_eff_list_err, fmt='.', color="xkcd:green",label="Signal Eff", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err, fmt='.', color="xkcd:denim",label="Background Eff", markersize='5')
plt.legend(loc='center right', title=f' NNcut = {NN_cut}\n Global signal eff ={sig_eff_global:.2f}\n Global bkg eff ={bkg_eff_global:.2f}')
if isBarrel == True :
    plt.title('Efficiencies in Pt bins (Barrel photons)', fontsize=20)
else :
    plt.title('Efficiencies in Pt bins (Endcap photons)', fontsize=20)
plt.xlabel('Pt bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-0.05,1.05)
plt.xlim(0,200)
plt.grid(axis="x")
plt.savefig(f'efficiency/' + modelname +'_'+ nn_cut + f'/eff_Pt_bins_{modelname}.png')
plt.close()

if isBarrel == True :
    #Eta bins:
    Eta_bins_err = np.diff(Eta_bins)
    Eta_bins_plot = []
    i = 0
    while i<len(Eta_bins_err):
        x_point = Eta_bins[i] + (Eta_bins_err[i]/2)
        Eta_bins_plot.append(x_point)
        i = i+1

else :
    Eta_bins_err1 = np.diff(Eta_bins1)
    Eta_bins_plot1 = []
    i = 0
    while i<len(Eta_bins_err1):
        x_point = Eta_bins1[i] + (Eta_bins_err1[i]/2)
        Eta_bins_plot1.append(x_point)
        i = i+1
        
    Eta_bins_err2 = np.diff(Eta_bins2)
    Eta_bins_plot2 = []
    i = 0
    while i<len(Eta_bins_err2):
        x_point = Eta_bins2[i] + (Eta_bins_err2[i]/2)
        Eta_bins_plot2.append(x_point)
        i = i+1
        
plt.figure(figsize=(8,8))
if isBarrel == True :
    plt.errorbar(Eta_bins_plot, sig_eff_eta, xerr = Eta_bins_err/2, yerr=sig_eff_eta_err, fmt='.', color="xkcd:green",label="Signal Eff", markersize='5')
    plt.errorbar(Eta_bins_plot, bkg_eff_eta, xerr = Eta_bins_err/2, yerr=bkg_eff_eta_err, fmt='.', color="xkcd:denim",label="Background Eff", markersize='5')
    plt.legend(loc='center right', title=f' NNcut = {NN_cut}\n Global signal eff ={sig_eff_global:.2f}\n Global bkg eff ={bkg_eff_global:.2f}')
    plt.title('Efficiencies in Eta bins (Barrel Photons) ', fontsize=20)
else :
    plt.errorbar(Eta_bins_plot1, sig_eff_eta1, xerr = Eta_bins_err1 / 2, yerr=sig_eff_eta_err1, fmt='.', color="xkcd:green",label="Signal Eff", markersize='5')
    plt.errorbar(Eta_bins_plot2, sig_eff_eta2, xerr = Eta_bins_err2 / 2, yerr=sig_eff_eta_err2, fmt='.', color="xkcd:green", markersize='5')
    plt.errorbar(Eta_bins_plot1, bkg_eff_eta1, xerr = Eta_bins_err1 / 2, yerr=bkg_eff_eta_err1, fmt='.', color="xkcd:denim",label="Background Eff", markersize='5')
    plt.errorbar(Eta_bins_plot2, bkg_eff_eta2, xerr = Eta_bins_err2 / 2, yerr=bkg_eff_eta_err2, fmt='.', color="xkcd:denim", markersize='5')
    plt.legend(loc='center right', title=f' NNcut = {NN_cut}\n Global signal eff ={sig_eff_global:.2f}\n Global bkg eff ={bkg_eff_global:.2f}')
    plt.title('Efficiencies in Eta bins (Endcap Photons) ', fontsize=20)
plt.xlabel('Eta bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-0.05,1.05)
plt.xlim(-3.1415,3.1415)
plt.grid(axis="x")
plt.savefig(f'efficiency/' + modelname +'_'+ nn_cut + f'/eff_Eta_bins_{modelname}.png')
plt.close()

txt.close()

print("\nSuccess!\n")
