#############################################################################
#                   TESTING A SEQUENTIAL NEURAL NETWORK                     #
#                                                                           #
# This code reads into testing CSV Files and the trained model.h5 file.     #
# It makes the evaluator ROC plot and identifies some working points.       #
# it should be run as follows :                                             #
#                                                                           #
#      python PFPhoton-ID-Evaluator.py <modelname> <barrel/endcap>          #
#                                                                           #
# NOTE : The max and minvalues are automatically taken from the models      #
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
txt = open(f'evaluated/' + modelname+ f'/info_{modelname}.txt', "w+")

##########################################################
#                    Settings:                           #
##########################################################

#Do you want to debug?
isDebug = False #True -> nrows=1000
#Do you want barrel or endcap?
if sys.argv[2] == 'barrel':
    isBarrel = True #True -> Barrel, False -> Endcap
elif sys.argv[2] == 'endcap':
    isBarrel = False
else:
    print('Please mention "barrel" or "endcap"')

train_var = ['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'phohasPixelSeed','phoR9Full5x5','phohcalTower']
#variables used in the training
varnames = ['hadTowOverEm', 'trkSumPtHollowConeDR03', 'ecalRecHitSumEtConeDR03','sigmaIetaIeta','SigmaIEtaIEtaFull5x5','SigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'hasPixelSeed','R9Full5x5','hcalTowerSumEtConeDR03']
#In the same order as they are fed into the training
#removed : 'phoEcalPFClusterIso','phoHcalPFClusterIso',

###################################################################################################################################################################

#READING DATA FILES :
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton

print('\nReading the input files.')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file5 = pd.read_csv('../TrainingSamples/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
else : #take all the photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=10000)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)
    file5 = pd.read_csv('../TrainingSamples/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)
    
##################################################################################################################################################################
#  Defining the Signal dataframes   #
#####################################

print('Defining Signal.')
#signal1 = file1[file1['isPhotonMatching'] ==1 ]
signal2 = file2[(file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 1) ]
#signal3 = file3[file3['isPhotonMatching'] ==1 ]
signal4 = file4[(file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 1) ]
signal5 = file5[(file5['isPhotonMatching'] == 1) & (file5['isPromptFinalState'] == 1) ]

#######################################
# Defining the Background data-frames #
#######################################

print('Defining Background.')
#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] ==0 ] 
background2 = file2[(file4['isPhotonMatching'] == 0) | ((file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 0)) ]
#background3 = file3[file3['isPhotonMatching'] ==0 ]
background4 = file4[(file4['isPhotonMatching'] == 0) | ((file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 0)) ]
background5 = file5[(file5['isPhotonMatching'] == 0) | ((file5['isPhotonMatching'] == 1) & (file5['isPromptFinalState'] == 0)) ]

#Note all signal/background photons from all the files are considered while evaluating.
##################################################################
#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
signal2["sample"]=1
#signal3["sample"]=2
signal4["sample"]=3
signal5["sample"]=4
#background1["sample"]=0
background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3
background5["sample"]=4

#signal1["label"]=1
signal2["label"]=1
#signal3["label"]=1
signal4["label"]=1
signal5["label"]=1
#background1["label"]=0
background2["label"]=0
#background3["label"]=0
background4["label"]=0
background5["label"]=0

################################################################
#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal2, signal4, signal5])
if isBarrel == True :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel only
else:
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap only

#Manually reducing signals :
Sig_alldf=Sig_alldf.sample(frac=1).reset_index(drop=True) #randomizing the rows 
Sig_alldf=Sig_alldf.head(1000000) #Keeps only the first 1 million rows

Bkg_alldf = pd.concat([background2, background4, background5])
if isBarrel == True :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel only
else :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap only

data = pd.concat([Sig_alldf, Bkg_alldf])
data_train, data_test = train_test_split(data, test_size=0.5, stratify=data["label"])
data=data_test #This makes sure that the testing samples are orthogonal
#If a completely new dataset is used, this step is not necessary.
print('Dataframes are succesfully created.')

##########################################################
n_sig = len(data.query('(label==1)'))
n_bkg = len(data.query('(label==0)'))

print(f"n_sig = {n_sig}")
print(f"n_bkg = {n_bkg}")
#Statistics:
txt.write("\n\nNUMBER OF PHOTONS :")
txt.write(f'\nTotal number of Signal Photons : {n_sig}')
txt.write(f'\nTotal number of Background Photons : {n_bkg}')
QCD_contribution = len(data.query('(label==0) & (sample==3)'))
Tau_contribution = len(data.query('(label==0) & (sample==4)'))
QCD_contribution_frac = (QCD_contribution*100) / (QCD_contribution+Tau_contribution)
Tau_contribution_frac = (Tau_contribution*100) / (QCD_contribution+Tau_contribution)
txt.write(f'\nContribution from QCD file = {QCD_contribution} ({QCD_contribution_frac:.0f}%)')
txt.write(f'\nContribution from TauGun file = {Tau_contribution} ({Tau_contribution_frac:.0f}%)')

X_sig, y_sig = Sig_alldf[train_var].values, Sig_alldf[['label']].values
X_bkg, y_bkg = Bkg_alldf[train_var].values, Bkg_alldf[['label']].values

#########################################################
#                     NORMALISATION                     #
#########################################################
# Reading from the scaler file:
maxValues=[]
minValues=[]

scaler_file=open(f'models/{modelname}/scaler_{modelname}.txt',"r")
lines=scaler_file.readlines()
for x in lines:
    maxValues.append(float(x.split(' ')[3]))
    minValues.append(float(x.split(' ')[2]))
scaler_file.close()
print('\nNormalisation Paremeters (min-max):')
print(f'maxValues = {maxValues}')
print(f'minValues = {minValues}')

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
print("The data has been normalised.\n")

#print(X_sig.shape)
#print(X_bkg.shape)

########################################################
#             Loading the neural network               #
########################################################

print('\nLoading the model.')
mymodel = tf.keras.models.load_model('models/'+ modelname + '/' + modelname + '.h5')
mymodel.load_weights('models/'+ modelname + '/' + modelname + '.h5')
print("Model loaded successfully.")
print('\nBEGINNING THE TESTING PROCESS\n')

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

X, y = data[train_var].values, data[['label']].values
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

data=data.drop(['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower', 'phoPhi', 'phohadTowOverEmValid', 'photrkSumPtSolid', 'isHardProcess'], axis=1)

print(data.head)
########################################################################
#test :
true_positive = data[(data['NNScore']>0.2) & (data['label']==1)]
manual_sigeff = (len(true_positive))/(len(data.query('(label == 1)')))
print('\nTest')
print(f'Manual sig_eff @cut0.2 is {manual_sigeff*100}\n')



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
NN_cuts=np.arange(0, 1.01, 0.01)
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
    
    if fnr_temp == matching_fnr :
        Optimum_Cut = NN_cuts[i]
        break
    i=i+1

print("\nTABLE FOR DIFFERENT NN_CUTS AND FNR :")
i=0
while i<len(NN_cuts):
    num_temp = len(data.query(f' (label==0) & (NNScore < {NN_cuts[i]})')) #bkg photons passing the NN cut
    den_temp = len(data.query(f' (label==0)')) #all bkg photons
    fnr_temp = (num_temp*100)/den_temp
    print(f"At NNCut = {NN_cuts[i]:.2f} fnr_NN ~ {fnr_temp:.2f}, fnr_PF ~ {fnr_PF:.2f}, difference from fnr_PF = {(fnr_PF-fnr_temp):.3f}")
    i=i+1
    


#REPORT :
print('\n############# Comparing NN to the PF-ID ###############')
print(f'For the same value of bkg_reg in the current PF-ID ({fnr_PF:.2f}), \nthe corresponding sig_eff is : {tpr_NN:.2f}')
print(f'Sig_eff has increased from {tpr_PF:.2f} to {tpr_NN:.2f}')
txt.write(f'\nFor the same value of bkg_reg in the current PF-ID ({fnr_PF:.2f}), \nthe corresponding sig_eff is : {tpr_NN:.2f}')
txt.write(f'\nSig_eff has increased from {tpr_PF:.2f} to {tpr_NN:.2f}')
print(f'\nThe required NN_cut is at = {Optimum_Cut:.2f}')
txt.write(f'\nThe required NN_cut is at = {Optimum_Cut:.2f}')
print('#######################################################')

#extra points above and below this cut:
def find_nearby(Cut, err):
    #tpr points:
    sig_num1=len(data.query(f' (label==1) & (NNScore > {Cut+err})'))
    sig_num2=len(data.query(f' (label==1) & (NNScore > {Cut-err})'))
    sig_den= len(data.query(f' (label==1)'))
    tpr1=(sig_num1*100)/sig_den
    tpr2=(sig_num2*100)/sig_den
    #fnr points:
    bkg_num1=len(data.query(f' (label==0) & (NNScore < {Cut+err})'))
    bkg_num2=len(data.query(f' (label==0) & (NNScore < {Cut-err})'))
    bkg_den= len(data.query(f' (label==0)'))
    fnr1=(bkg_num1*100)/bkg_den
    fnr2=(bkg_num2*100)/bkg_den

    print(Cut+err)
    print(err)
    #return :
    if (Cut+err)<1 and (Cut-err)>0 :
        return tpr1, fnr1, tpr2, fnr2
    else :
        print("Error :The cuts are beyond the limit (0, 1)")
        print("Fix the err variable.")

#We also select two working points above and below the optimum cut, which are defined as follows.
err = 0.1
if Optimum_Cut > 0.92:
    err = abs(1-Optimum_Cut) - 0.005
    
tpr1, fnr1, tpr2, fnr2 = find_nearby(Optimum_Cut, err)

    
########################################################
#                      Plotting                        #
########################################################

print('\nPlotting has begun')
#Plotting the ROC:
plt.figure(figsize=(8,8))
#ROC:
plt.plot(mytpr,myfnr,color='xkcd:bright blue',label='Testing AUC = %0.4f' % myauc)
#CMSSW flag:
plt.plot(signaleff, backgroundrej, marker='o', color="red", markersize=8, label=f'CMSSW flag ({signaleff:.0f},{backgroundrej:.0f})')
#Three points on the ROC
print(tpr1)
print(fnr1)
plt.xlim(0,100)
plt.ylim(0,100)
plt.plot(tpr1, fnr1, marker=(5, 1, 0),  color="blue", markersize=8, label=f'NN cut at {(Optimum_Cut+err):.2f} ({tpr1:.0f},{fnr1:.0f})')
plt.plot(tpr_NN, fnr_NN, marker=(5, 1, 0), color="red", markersize=8, label=f'NN cut at {Optimum_Cut:.2f} ({tpr_NN:.0f},{fnr_NN:.0f})')
plt.plot(tpr2, fnr2, marker=(5, 1, 0), color="black", markersize=8, label=f'NN cut at {(Optimum_Cut-err):.2f} ({tpr2:.0f},{fnr2:.0f})')
#marker=(5, 1, 0),

plt.legend(loc='lower right')
if isBarrel == True :
    plt.title(f'ROC curve (testing, barrel)',fontsize=20)
else :
    plt.title(f'ROC curve (testing, endcap)',fontsize=20)
plt.xlabel('Signal Efficiency',fontsize=20)
plt.ylabel('Background Rejection',fontsize=20)
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
