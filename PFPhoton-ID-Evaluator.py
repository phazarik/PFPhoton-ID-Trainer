#############################################################################
#                   TESTING A SEQUENTIAL NEURAL NETWORK                     #
#                                                                           #
# This code reads into testing CSV Files and the trained model.h5 file.     #
# It makes the evaluator ROC plot.                                          #
# it should be run as follows :                                             #
#            python PFPhoton-ID-Evaluator.py <trainedmodelname>             #
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
os.system("mkdir -p evaluated/" + modelname)
txt = open(f'evaluated/' + modelname+ f'/Info_{modelname}.txt', "w+")

#Do you want barrel or endcap?
isBarrel = True #True -> Barrel, False -> Endcap
#Do you want to debug?
isDebug = False #True -> nrows=1000

###################################################################################################################################################################

#READING DATA FILES :
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton

print('Reading the input files')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    #file5 = pd.read_csv('../TrainingSamples/df_TauGun_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
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
data = pd.concat([Sig_alldf, Bkg_alldf])

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
print("\nThe data has been normalised.\n")

########################################################
#             Loading the neural network               #
########################################################

mymodel = tf.keras.models.load_model('output/'+ modelname + '/' + modelname + '.h5')
mymodel.load_weights('output/'+ modelname + '/' + modelname + '.h5')
print("model loaded")

#Caclculating tpr, fnr and adding the NN score as a separate column to the dataframe:
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

###################################################################
# The Current PF-ID :
backgroundpass=len(data.query('(isPFPhoton == 1) & (label == 0)'))
backgroundrej =len(data.query('(isPFPhoton == 0) & (label == 0)'))
signalpass    =len(data.query('(isPFPhoton == 1) & (label == 1)'))
signalrej     =len(data.query('(isPFPhoton == 0) & (label == 1)'))
backgroundrej =( backgroundrej/(backgroundpass+backgroundrej) )*100
signaleff     =( signalpass/(signalpass+signalrej) )*100


########################################################
#    Calculating tpr, fnr and NNscore from a fnr_PF    #
#     (Background Rejection of the current PF-ID):     #
########################################################
# We have a dataframe called 'data' which contains signal and background photons,
# each of which has a NN score. We get that from the model as follows.
v_df = pd.DataFrame()
v_df['truth'] = data['label'].values
v_df['prob']=0

X, y = data[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values, data[['label']].values
normedX = 2*((X - minValues)/(MaxMinusMin)) -1.0
X = normedX

val_pred_proba = mymodel.predict(X)
#v_df['prob'] = val_pred_proba
# The NN score for each photon is given by -
#v_df[v_df['test_truth']==1]['test_prob'] #(signal photons) and
#v_df[v_df['test_truth']==0]['test_prob'] #(background photons)

#########################################################################
#We store the NNScore in a new column :
print('\nAdding the NN score to the dataframe.')
print(data.shape)
data["NNScore"] = val_pred_proba
#data.loc[data['label'] == 1, "NNScore"] =  v_df[v_df['truth']==1]['prob']
#data.loc[data['label'] == 0, "NNScore"] =  v_df[v_df['truth']==0]['prob']
print(f'NN score added. The dataframe looks like - ')
print(data.shape)

data=data.drop(['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower', 'phoPt', 'phoEta', 'phoPhi', 'phohadTowOverEmValid', 'photrkSumPtSolid', 'isHardProcess'], axis=1)

print(data.head)
#########################################################################

#The following two qunatites are sig_eff (tpr) and bkg_rej (fnr) of the current PF-ID.
tpr_PF = signaleff
fnr_PF = backgroundrej

#matching_function :
#Given a particular value, it finds the closest value in a list, along with the index. 
def match(value_, list_) :
    i=0
    diff=[]
    #step 1 finding the minimum difference :
    while i < len(list_):
        diff_value = abs(value_ - list_[i])
        diff.append(diff_value)
        i = i+1
    min_diff = min(diff)
    j=0
    while j < len(list_):
        diff_value = abs(value_ - list_[j])
        if diff_value == min_diff:
            return list_[j], j #We also keep the matching index
        j = j+1

# For a particular background rejection (fnr_PF), we need the matching tpr_NN, fpr_NN and the corresponding NN cut.

fnr_NN, index = match(fnr_PF, myfnr)
tpr_NN = mytpr[index]

#Estimating the cut on the NN score that gives tpr_NN and fpr_NN:
NN_cuts=np.arange(0.7, 1, 0.005)
fnr_cuts=[]
print(f'Trial cuts on the NN score = {NN_cuts}')
print(f'Target fnr = {fnr_NN}')
i=0
while i<len(NN_cuts):
    num_temp = len(data.query(f' (label==0) & ( NNScore < {NN_cuts[i]} )')) #bkg photons passing the NN cut
    den_temp = len(data.query(f' (label==0)')) #all bkg photons
    fnr_temp = (num_temp*100)/den_temp
    fnr_cuts.append(fnr_temp)
    i=i+1

print('\nHit and trial fnr =')
print(fnr_cuts)

matching_fnr, index = match(fnr_NN, fnr_cuts)
print(f'\nmatching fnr = {matching_fnr}')

i=0
while i<len(NN_cuts):
    num_temp = len(data.query(f' (label==0) & (NNScore < {NN_cuts[i]})')) #bkg photons passing the NN cut
    den_temp = len(data.query(f' (label==0)')) #all bkg photons
    fnr_temp = (num_temp*100)/den_temp
    #print('')
    #print(fnr_temp)
    #print(abs(fnr_temp - matching_fnr))
    if fnr_temp == matching_fnr :
        Optimum_Cut = NN_cuts[i]
        break
    i=i+1


#REPORT :
print('\n############# Comparing NN to the PF-ID ###############')
print(f'For the same value of bkg_reg in the current PF-ID ({fnr_PF:.2f}), \nthe corresponding sig_eff is : {tpr_NN:.2f}')
print(f'Sig_eff has increased from {tpr_PF:.2f} to {tpr_NN:.2f}')
txt.write(f'\nFor the same value of bkg_reg in the current PF-ID ({fnr_PF:.2f}), \nthe corresponding sig_eff is : {tpr_NN:.2f}')
txt.write(f'\nSig_eff has increased from {tpr_PF:.2f} to {tpr_NN:.2f}')
print(f'\nThe required NN_cut is at = {NN_cuts[i]:.2f}')
txt.write(f'\nThe required NN_cut is at = {NN_cuts[i]:.2f}')
print('#######################################################')
    
########################################################
#                      Plotting                        #
########################################################

print('\nPlotting has begun')
#Plotting the ROC:
plt.figure(figsize=(8,8))
plt.plot([signaleff], [backgroundrej], marker='o', color="red", markersize=6, label='CMSSW flag')
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

# NN score histogram :
plt.figure(figsize=(8,6))
mybins = np.arange(0, 1.02, 0.02)
plt.hist(data[data['label']==1]['NNScore'], bins=mybins, histtype='step', density=False, color="xkcd:green",label="Signal Photons", linewidth=2)
plt.hist(data[data['label']==0]['NNScore'], bins=mybins, histtype='step', density=False, color="xkcd:denim",label="Background Photons", linewidth=2)
plt.legend(loc='upper center')
if isBarrel == True :
    plt.title(f'NN score plot (testing, barrel)',fontsize=20)
else :
    plt.title(f'NN score plot (testing, endcap)',fontsize=20)
plt.xlabel('NN score',fontsize=20)
plt.ylabel('No. of Photons',fontsize=20)
plt.yscale("log")
plt.xlim(0,1)
plt.savefig('evaluated/' + modelname + '/' + modelname + '_NN.png')
plt.close()


print(f'\nAll done. Evaluated ROC is saved in the folder : evaluated/{modelname}\n')
txt.close()
print("Success!\n")
