#############################################################################
#                             EFFICIENCY CHECKER                            #
#                                                                           #
# Given a NN_cut, this code reads into CSV Files, and plots signal and      #
# background efficiencies for that cut in different pT bins.                #
# It should be run as follows :                                             #
#                                                                           #
#  python PFPhoton-ID-Efficiency.py <modelname> <barrel/endcap> <NNCut>     #
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
import math 

###################################################################################################################################################################
os.system("")
modelname = sys.argv[1]
nn_cut = float(sys.argv[3])
os.system("mkdir -p efficiency/" + modelname + '_' + str(nn_cut))
txt = open(f'efficiency/' + modelname +'_'+ str(nn_cut) + f'/Info_{modelname}.txt', "w+")

#Cut on the Neural Network plot :
NN_cut = float(nn_cut)

##########################################################
#                    Settings:                           #
##########################################################
#Do you want normalised plot?
isNorm = True
#Do you want to debug?
isDebug = False #True -> nrows=1000

#Do you want barrel or endcap?
if sys.argv[2] == 'barrel':
    isBarrel = True #True -> Barrel, False -> Endcap
elif sys.argv[2] == 'endcap':
    isBarrel = False
else:
    print('Please mention "barrel" or "endcap"')


train_var = ['phoHoverE', 'photrkSumPtHollow','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'phohasPixelSeed','phoR9Full5x5']
#variables used in the training
varnames = ['hadTowOverEm', 'trkSumPtHollowConeDR03','sigmaIetaIeta','SigmaIEtaIEtaFull5x5','SigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'hasPixelSeed','R9Full5x5']
#In the same order as they are fed into the training
#removed : 'phoEcalPFClusterIso','phoHcalPFClusterIso',

##############################################################################
#READING DATA FILES :
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton

print('Reading the input files')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file2 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file4 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file5 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
else : #take all the photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols)
    file2 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    file4 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)
    file5 = pd.read_csv('../TrainingSamples/Run3Summer21-v2/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)

#########################################################
#                  making dataframes                    #
#########################################################
print('1. Defining Signal.')
#signal1 = file1[(file1['isPhotonMatching'] == 1) and (file1['isPromptFinalState'] == 1) ]
signal2 = file2[(file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 1) ]
#signal3 = file3[(file3['isPhotonMatching'] == 1) and (file3['isPromptFinalState'] == 1)]
print('Done.')

print('2. Defining Background.')
#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] == 0 ] 
#background2 = file2[file2['isPhotonMatching'] == 0 ]
#background3 = file3[file3['isPhotonMatching'] == 0 ]
background4 = file4[(file4['isPhotonMatching'] == 0) | ((file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 0)) ]
background5 = file5[(file5['isPhotonMatching'] == 0) | ((file5['isPhotonMatching'] == 1) & (file5['isPromptFinalState'] == 0)) ]
print('Done.')

#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
signal2["sample"]=1
#signal3["sample"]=2
#background1["sample"]=0
#background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3
background5["sample"]=4

#signal1["label"]=1
signal2["label"]=1
#signal3["label"]=1
#background1["label"]=0
#background2["label"]=0
#background3["label"]=0
background4["label"]=0
background5["label"]=0

#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal2])

if isBarrel == True :
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel only
else:
    Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap only

#Manually reducing signals :
Sig_alldf=Sig_alldf.sample(frac=1).reset_index(drop=True) #randomizing the rows 
#Sig_alldf=Sig_alldf.head(1000000) #Keeps only the first 1 million rows

print('\nShape of the dataframes :')
print(f'Signal : {Sig_alldf.shape}')

Bkg_alldf = pd.concat([background4, background5])
if isBarrel == True :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) < 1.442] #barrel only
else :
    Bkg_alldf = Bkg_alldf[abs(Bkg_alldf['phoEta']) > 1.566] #endcap only

print(f'Background : {Bkg_alldf.shape}')

#Final data frame creaton :    
data = pd.concat([Sig_alldf,Bkg_alldf])

print('\nData reading succesful!')
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


#X_sig, y_sig = Sig_alldf[train_var].values, Sig_alldf[['label']].values
#X_bkg, y_bkg = Bkg_alldf[train_var].values, Bkg_alldf[['label']].values
#We don't need to split it, since we are only evaluating the NN

#########################################################
#                     NORMALISATION                     #
#########################################################
#The following two lists are to be manually put from the training step.

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
    
#normedX_sig = 2*((X_sig - minValues)/(MaxMinusMin)) -1.0
#X_sig = normedX_sig
#normedX_bkg = 2*((X_bkg - minValues)/(MaxMinusMin)) -1.0
#X_bkg = normedX_bkg

X, y = data[train_var].values, data[['label']].values
normedX = 2*((X - minValues)/(MaxMinusMin)) -1.0
X = normedX
print("The data has been normalised.")

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

#myfnr, mytpr, myauc = get_roc_details(mymodel,X_sig,y_sig,X_bkg,y_bkg)

v_df = pd.DataFrame()
v_df['truth'] = data['label'].values
v_df['prob']=0

val_pred_proba = mymodel.predict(X)

#########################################################################
#We store the NNScore in a new column :
print('\nAdding the NN score to the dataframe.')
print(data.shape)
data["NNscore"] = val_pred_proba
#data.loc[data['label'] == 1, "NNScore"] =  v_df[v_df['truth']==1]['prob']
#data.loc[data['label'] == 0, "NNScore"] =  v_df[v_df['truth']==0]['prob']
print(f'NN score added. The dataframe looks like - ')
print(data.shape)

data=data.drop(['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5', 'phohasPixelSeed','phoR9Full5x5','phohcalTower', 'phoPhi', 'phohadTowOverEmValid', 'photrkSumPtSolid', 'isHardProcess'], axis=1)

print(data.head)

##########################################################################
#At this point, the dataframe 'data' has all the photons wth their NNscore
#Efficiency calculation starts here.
##########################################################################



######################################
#        defining the Pt bins        #
######################################
print('\nBinning for efficiency plot.')
Pt_bins=[10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
print(f'Number of Pt bins = {len(Pt_bins)-1}')
data['Pt_bin'] =  pd.cut(data['phoPt'], bins=Pt_bins, labels=list(range(len(Pt_bins)-1)))
#This cuts the data into different bins of Pt with labels 0, 1 , 2 ... etc

Pt_bins_err = np.diff(Pt_bins)
Pt_bins_plot = []
i = 0
while i<len(Pt_bins_err):
    x_point = Pt_bins[i] + (Pt_bins_err[i]/2)
    Pt_bins_plot.append(x_point)
    i = i+1
#On the plot, the midpoints of the Pt bins are plotted.

#######################################
#      efficiency calculations        # 
#######################################

'''
def efficiency(label_, bintype_, binno_, nn_) :
    numerator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_}) & (NNscore > {nn_})'))
    denominator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        print('Some bins have 0 entries, setting eff to zero')
        return 0, 0
    else :
        eff = (numerator*100)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err

def efficiency_sample(sample_, label_, bintype_, binno_, nn_) :
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (NNscore > {nn_})'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = (numerator*100)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err

def efficiency_PF(sample_, label_, bintype_, binno_, boolPF_):
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (isPFPhoton == {boolPF_})'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = (numerator*100)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err
'''

#########################
#        Global         #
#########################

#calculatin of global efficiencies at NN_cut (irrespective of bins) :

sig_eff_global=0
bkg_eff_global=0

num_sig = len(data.query(f'(label == 1) &(NNscore > {NN_cut})'))
den_sig = len(data.query(f'(label == 1)'))
num_bkg = len(data.query(f'(label == 0) & (NNscore > {NN_cut})'))
den_bkg = len(data.query(f'(label == 0)'))
if(den_sig > 0):
    sig_eff_global = (num_sig*100)/den_sig
if(den_bkg > 0):
    bkg_eff_global = (num_bkg*100)/den_bkg

'''
#In Pt Bins:
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
'''

###########################
#  For different samples  #
###########################

'''
def global_efficiency_sample(sample_) :
    sig_eff_global1 = 0
    bkg_eff_global1 = 0
    num_sig1 = len(data.query(f'(sample == {sample_}) & (label == 1) &(NNscore > {NN_cut})'))
    den_sig1 = len(data.query(f'(sample == {sample_}) &(label == 1)'))
    num_bkg1 = len(data.query(f'(sample == {sample_}) &(label == 0) & (NNscore > {NN_cut})'))
    den_bkg1 = len(data.query(f'(sample == {sample_}) &(label == 0)'))
    if(den_sig1 > 0):
        sig_eff_global1 = (num_sig1)*100/den_sig1
    if(den_bkg1 > 0):
        bkg_eff_global1 = (num_bkg1)*100/den_bkg1
    return sig_eff_global1, bkg_eff_global1

sig_eff_global_GJet, bkg_eff_global_GJet = global_efficiency_sample(1)
sig_eff_global_QCD, bkg_eff_global_QCD = global_efficiency_sample(3)
sig_eff_global_Tau, bkg_eff_global_Tau = global_efficiency_sample(4)
'''
###
##################################################################################
#MAIN
#In Pt Bins :
def efficiency_sample(sample_, label_, bintype_, binno_, nn_) :
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (NNscore > {nn_})'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = numerator/denominator
        eff_err = math.sqrt(eff*(1-eff))/denominator
        eff = eff*100
        return eff, eff_err
    
def calculate_pt_eff_sample(sample_):
    sig_eff_list1 = []
    sig_eff_list_err1 =[]
    bkg_eff_list1 = []
    bkg_eff_list_err1 = []
    for iterator in range(len(Pt_bins)-1):
        sig_eff1, sig_eff_err1 = efficiency_sample(sample_, 1, 'Pt_bin', iterator, NN_cut)
        bkg_eff1, bkg_eff_err1 = efficiency_sample(sample_, 0, 'Pt_bin', iterator, NN_cut)
        sig_eff_list1.append(sig_eff1)
        bkg_eff_list1.append(bkg_eff1)
        sig_eff_list_err1.append(sig_eff_err1)
        bkg_eff_list_err1.append(bkg_eff_err1)
        
    return sig_eff_list1, sig_eff_list_err1, bkg_eff_list1, bkg_eff_list_err1

sig_eff_list_GJet, sig_eff_list_err_GJet, bkg_eff_list_GJet, bkg_eff_list_err_GJet = calculate_pt_eff_sample(1)
sig_eff_list_QCD, sig_eff_list_err_QCD, bkg_eff_list_QCD, bkg_eff_list_err_QCD = calculate_pt_eff_sample(3)
sig_eff_list_Tau, sig_eff_list_err_Tau, bkg_eff_list_Tau, bkg_eff_list_err_Tau = calculate_pt_eff_sample(4)
########################################################################################################
#Same Calculations for PF:
#MAIN
#In Pt Bins :
def efficiency_sample_PF(sample_, label_, bintype_, binno_, boolPF_) :
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (isPFPhoton == {boolPF_})'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = numerator/denominator
        eff_err = math.sqrt(eff*(1-eff))/denominator
        eff = eff*100
        return eff, eff_err

def calculate_pt_eff_sample_PF(sample_):
    sig_eff_list1 = []
    sig_eff_list_err1 =[]
    bkg_eff_list1 = []
    bkg_eff_list_err1 = []
    for iterator in range(len(Pt_bins)-1):
        sig_eff1, sig_eff_err1 = efficiency_sample_PF(sample_, 1, 'Pt_bin', iterator, 1)
        bkg_eff1, bkg_eff_err1 = efficiency_sample_PF(sample_, 0, 'Pt_bin', iterator, 1)
        sig_eff_list1.append(sig_eff1)
        bkg_eff_list1.append(bkg_eff1)
        sig_eff_list_err1.append(sig_eff_err1)
        bkg_eff_list_err1.append(bkg_eff_err1)

    return sig_eff_list1, sig_eff_list_err1, bkg_eff_list1, bkg_eff_list_err1


sig_eff_list_GJet_PF, sig_eff_list_err_GJet_PF, bkg_eff_list_GJet_PF, bkg_eff_list_err_GJet_PF = calculate_pt_eff_sample_PF(1)
sig_eff_list_QCD_PF, sig_eff_list_err_QCD_PF, bkg_eff_list_QCD_PF, bkg_eff_list_err_QCD_PF = calculate_pt_eff_sample_PF(3)
sig_eff_list_Tau_PF, sig_eff_list_err_Tau_PF, bkg_eff_list_Tau_PF, bkg_eff_list_err_Tau_PF = calculate_pt_eff_sample_PF(4)

#########################################
#               plotting                #
#########################################
location = 'center right'
bins_= np.arange(0, 151, 1)

sig_eff_PF = (  len(data.query(f'(label == 1) &(isPFPhoton == 1)')) )*100/(  len(data.query(f'(label == 1)')) )
bkg_eff_PF = (  len(data.query(f'(label == 0) &(isPFPhoton == 1)')) )*100/(  len(data.query(f'(label == 0)')) )

#Plot 1 : pT efficiency plot (NN)
plt.figure(figsize=(8,8))
plt.errorbar(Pt_bins_plot, sig_eff_list_GJet, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_GJet, fmt='.', color="xkcd:green",label="Signal from GJet", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_QCD, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_QCD, fmt='.', color="xkcd:denim",label="Background from QCD", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_Tau, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_Tau, fmt='.', color="xkcd:red",label="Background from TauGun", markersize='5')
#plt.errorbar(Pt_bins_plot, sig_eff_list_QCD, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_QCD, fmt='.', color="xkcd:denim",label="QCD", markersize='5')
#plt.errorbar(Pt_bins_plot, sig_eff_list_Tau, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_Tau, fmt='.', color="xkcd:red",label="TauGun", markersize='5')
plt.legend(loc=location, title=f' NN cut = {NN_cut}\n Global signal eff ={sig_eff_global:.2f}%\n Global bkg eff ={bkg_eff_global:.2f}%')
if isBarrel == True :
    plt.title(f'Photon Efficiencies at NN cut = {NN_cut} (Barrel)', fontsize=15)
else :
    plt.title(f'Photon Efficiencies at NN cut = {NN_cut} (Endcap)', fontsize=15)
plt.xlabel('Pt bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-5,105)
plt.xlim(0,200)
plt.grid(axis="x")
plt.grid(axis="y")
plt.savefig(f'efficiency/' + modelname +'_'+ str(nn_cut) + f'/eff_Pt_bins_{modelname}.png')
plt.close()

#Plot 1 : pT efficiency plot (PF)
plt.figure(figsize=(8,8))
plt.errorbar(Pt_bins_plot, sig_eff_list_GJet_PF, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_GJet_PF, fmt='.', color="xkcd:green",label="Signal from GJet", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_QCD_PF, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_QCD_PF, fmt='.', color="xkcd:denim",label="Background from QCD", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_Tau_PF, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_Tau_PF, fmt='.', color="xkcd:red",label="Background from TauGun", markersize='5')
plt.legend(loc=location, title=f' CMSSW flag\n Global signal eff ={sig_eff_PF:.2f}%\n Global bkg eff ={bkg_eff_PF:.2f}%')
if isBarrel == True :
    plt.title('Photon Efficiencies for the CMSSW flag (Barrel)', fontsize=15)
else :
    plt.title('Photon Efficiencies for the CMSSW flag (Endcap)', fontsize=15)
plt.xlabel('Pt bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-5,105)
plt.xlim(0,200)
plt.grid(axis="x")
plt.grid(axis="y")
plt.savefig(f'efficiency/' + modelname +'_'+ str(nn_cut) + f'/eff_Pt_bins_PF_{modelname}.png')
plt.close()


#################################################################################
txt.close()
print(f'\nAll done. Plots are saved in the folder : efficiency/{modelname}_{nn_cut}\n')
