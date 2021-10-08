#############################################################################
#                        PF EFFICIENCY CHECKER                              #
#                                                                           #
# This code reads into CSV Files, and checks the efficiencies of CMSSW ID   #
# in differnt pT bins.                                                      #
# it should be run as follows :                                             #
#                                                                           #
#             python PFPhoton-ID-Efficiency_PF <barrel/endcap>              #
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
os.system(f"mkdir -p PFefficiency/" + sys.argv[1])
txt = open(f'PFefficiency/' + sys.argv[1] + f'/Info_{sys.argv[1]}.txt', "w+")

##########################################################
#                    Settings:                           #
##########################################################
#Do you want normalised plots?
isNorm = True
#Do you want to debug?
isDebug = True #True -> nrows=1000

#Do you want barrel or endcap?
if sys.argv[1] == 'barrel':
    isBarrel = True #True -> Barrel, False -> Endcap
elif sys.argv[1] == 'endcap':
    isBarrel = False
else:
    print('Please mention "barrel" or "endcap"')

train_var = ['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'phohasPixelSeed','phoR9Full5x5','phohcalTower']
#variables used in the training
varnames = ['hadTowOverEm', 'trkSumPtHollowConeDR03', 'ecalRecHitSumEtConeDR03','sigmaIetaIeta','SigmaIEtaIEtaFull5x5','SigmaIEtaIPhiFull5x5', 'phoEcalPFClusterIso','phoHcalPFClusterIso', 'hasPixelSeed','R9Full5x5','hcalTowerSumEtConeDR03']
#In the same order as they are fed into the training
#removed : 'phoEcalPFClusterIso','phoHcalPFClusterIso',

##############################################################################
#READING DATA FILES :
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton

print('Reading the input files')
mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
if isDebug == True : #take only the first 1000 photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
    file5 = pd.read_csv('../TrainingSamples/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=1000)
else : #take all the photons
    #file1 = pd.read_csv('../TrainingSamples/df_GJet_20to40_20.csv.gzip',compression='gzip', usecols=mycols)
    file2 = pd.read_csv('../TrainingSamples/df_GJet.csv.gzip',compression='gzip', usecols=mycols, nrows=1000000)
    #file3 = pd.read_csv('../TrainingSamples/df_GJet_40toInf_20.csv.gzip',compression='gzip', usecols=mycols)
    file4 = pd.read_csv('../TrainingSamples/df_QCD.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)
    file5 = pd.read_csv('../TrainingSamples/df_TauGun.csv.gzip',compression='gzip', usecols=mycols, nrows=250000)

#########################################################
#                  making dataframes                    #
#########################################################
print('creating the dataframe')
file2["sample"]=1
file4["sample"]=3
file5["sample"]=4
df=pd.concat([file2, file4, file5])
df=df.sample(frac=1).reset_index(drop=True) #randomizing the rows 
df=df.head(1500000)

print('defining signal')
sigdf = df[(df['isPhotonMatching'] == 1) & (df['isPromptFinalState'] == 1) ]
sigdf['label']=1

print('defining background')
bkgdf = df[(df['isPhotonMatching'] == 0) | ((df['isPhotonMatching'] == 1) & (df['isPromptFinalState'] == 0)) ]
bkgdf['label']=0

print('concatinating')
data=pd.concat([sigdf, bkgdf])
print(data.shape)

#Barrel/Endcap selection :
if isBarrel == True :
    data = data[abs(data['phoEta']) < 1.442] #barrel only
else:
    data = data[abs(data['phoEta']) > 1.566]
    
#splitting:
data_train, data_test = train_test_split(data, test_size=0.5, stratify=data["label"])
data=data_test #This makes sure that the testing samples are orthogonal
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


##########################################################################
#At this point, the dataframe 'data' has all the photons wth their NNscore
#Efficiency calculation starts here.
##########################################################################



######################################
#        defining the Pt bins        #
######################################
print('\nBinning for efficiency plot.')
Pt_bins=[5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
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

def efficiency(label_, bintype_, binno_) :
    numerator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_}) & (isPFPhoton == 1)'))
    denominator = len(data.query(f'(label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        print('Some bins have 0 entries, setting eff to zero')
        return 0, 0
    else :
        eff = (numerator)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err

def efficiency_sample(sample_, label_, bintype_, binno_) :
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (isPFPhoton == 1)'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = (numerator)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err

#########################
#  For all the samples  #
#########################

#calculatin of global efficiencies at NN_cut (irrespective of bins) :
sig_eff_global=0
bkg_eff_global=0

num_sig = len(data.query(f'(label == 1) &(isPFPhoton == 1)'))
den_sig = len(data.query(f'(label == 1)'))
num_bkg = len(data.query(f'(label == 0) & (isPFPhoton == 1)'))
den_bkg = len(data.query(f'(label == 0)'))
if(den_sig > 0):
    sig_eff_global = (num_sig)/den_sig
if(den_bkg > 0):
    bkg_eff_global = (num_bkg)/den_bkg

#In Pt Bins:
sig_eff_list = []
sig_eff_list_err =[]
bkg_eff_list = []
bkg_eff_list_err = []
for iterator in range(len(Pt_bins)-1):
    sig_eff, sig_eff_err = efficiency(1, 'Pt_bin', iterator)
    bkg_eff, bkg_eff_err = efficiency(0, 'Pt_bin', iterator)
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

###########################
#  For different samples  #
###########################

def global_efficiency_sample(sample_) :
    sig_eff_global1 = 0
    bkg_eff_global1 = 0
    num_sig1 = len(data.query(f'(sample == {sample_}) & (label == 1) & (isPFPhoton == 1)'))
    den_sig1 = len(data.query(f'(sample == {sample_}) &(label == 1)'))
    num_bkg1 = len(data.query(f'(sample == {sample_}) &(label == 0) & (isPFPhoton == 1)'))
    den_bkg1 = len(data.query(f'(sample == {sample_}) &(label == 0)'))
    if(den_sig1 > 0):
        sig_eff_global1 = (num_sig1)/den_sig1
    if(den_bkg1 > 0):
        bkg_eff_global1 = (num_bkg1)/den_bkg1
    return sig_eff_global1, bkg_eff_global1

sig_eff_global_GJet, bkg_eff_global_GJet = global_efficiency_sample(1)
sig_eff_global_QCD, bkg_eff_global_QCD = global_efficiency_sample(3)
sig_eff_global_Tau, bkg_eff_global_Tau = global_efficiency_sample(4)

#In Pt Bins :
def efficiency_sample(sample_, label_, bintype_, binno_) :
    numerator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_}) & (isPFPhoton == 1)'))
    denominator = len(data.query(f'(sample == {sample_}) & (label == {label_}) & ({bintype_} == {binno_})'))
    if denominator == 0 :
        return 0, 0
    else :
        eff = (numerator)/(denominator)
        eff_err = 1/math.sqrt(denominator)
        return eff, eff_err
    
def calculate_pt_eff_sample(sample_):
    sig_eff_list1 = []
    sig_eff_list_err1 =[]
    bkg_eff_list1 = []
    bkg_eff_list_err1 = []
    for iterator in range(len(Pt_bins)-1):
        sig_eff1, sig_eff_err1 = efficiency_sample(sample_, 1, 'Pt_bin', iterator)
        bkg_eff1, bkg_eff_err1 = efficiency_sample(sample_, 0, 'Pt_bin', iterator)
        sig_eff_list1.append(sig_eff1)
        bkg_eff_list1.append(bkg_eff1)
        sig_eff_list_err1.append(sig_eff_err1)
        bkg_eff_list_err1.append(bkg_eff_err1)
        
    return sig_eff_list1, sig_eff_list_err1, bkg_eff_list1, bkg_eff_list_err1

sig_eff_list_GJet, sig_eff_list_err_GJet, bkg_eff_list_GJet, bkg_eff_list_err_GJet = calculate_pt_eff_sample(1)
sig_eff_list_QCD, sig_eff_list_err_QCD, bkg_eff_list_QCD, bkg_eff_list_err_QCD = calculate_pt_eff_sample(3)
sig_eff_list_Tau, sig_eff_list_err_Tau, bkg_eff_list_Tau, bkg_eff_list_err_Tau = calculate_pt_eff_sample(4)


#########################################
#               plotting                #
#########################################
location = 'center right'
bins_= np.arange(0, 151, 1)

#Plot 1 : pT efficiency plot (Signal)
plt.figure(figsize=(8,8))
plt.errorbar(Pt_bins_plot, sig_eff_list_GJet, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_GJet, fmt='.', color="xkcd:green",label="GJet", markersize='5')
plt.errorbar(Pt_bins_plot, sig_eff_list_QCD, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_QCD, fmt='.', color="xkcd:denim",label="QCD", markersize='5')
plt.errorbar(Pt_bins_plot, sig_eff_list_Tau, xerr = Pt_bins_err/2, yerr=sig_eff_list_err_Tau, fmt='.', color="xkcd:red",label="TauGun", markersize='5')
plt.legend(loc=location, title=f' Global signal eff ={sig_eff_global:.2f}')
if isBarrel == True :
    plt.title('Signal Efficiencies in Pt bins (Barrel)', fontsize=20)
else :
    plt.title('Signal Efficiencies in Pt bins (Endcap)', fontsize=20)
plt.xlabel('Pt bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-0.05,1.05)
plt.xlim(0,200)
plt.grid(axis="x")
plt.savefig(f'PFefficiency/' + sys.argv[1] + f'/sig_eff_Pt_bins_{sys.argv[1]}.png')
plt.close()

#Plot 2 : pT efficiency plot (Background)
plt.figure(figsize=(8,8))
plt.errorbar(Pt_bins_plot, bkg_eff_list_GJet, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_GJet, fmt='.', color="xkcd:green",label="GJet", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_QCD, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_QCD, fmt='.', color="xkcd:denim",label="QCD", markersize='5')
plt.errorbar(Pt_bins_plot, bkg_eff_list_Tau, xerr = Pt_bins_err/2, yerr=bkg_eff_list_err_Tau, fmt='.', color="xkcd:red",label="TauGun", markersize='5')
plt.legend(loc=location, title=f' Global bkg eff ={bkg_eff_global:.2f}')
if isBarrel == True :
    plt.title('Background Efficiencies in Pt bins (Barrel)', fontsize=20)
else :
    plt.title('Background Efficiencies in Pt bins (Endcap)', fontsize=20)
plt.xlabel('Pt bins',fontsize=20)
plt.ylabel('Efficiency',fontsize=15)
plt.ylim(-0.05,1.05)
plt.xlim(0,200)
plt.grid(axis="x")
plt.savefig(f'PFefficiency/' + sys.argv[1] + f'/bkg_eff_Pt_bins_{sys.argv[1]}.png')
plt.close()

#plot 3 : pT spectrum (signal):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 1) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False)
plt.hist(data.query('(label == 1) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False)
plt.hist(data.query('(label == 1) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False)
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Signal Photon Pt (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Signal Photon Pt (endcap)',fontsize=20) #endcap only
plt.savefig(f'PFefficiency/' + sys.argv[1]  +f'/phoPt_{sys.argv[1]}_sigonly.png')
plt.close()

#plot 4 : pT spectrum (background):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 0) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False)
plt.hist(data.query('(label == 0) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False)
plt.hist(data.query('(label == 0) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False)
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Background Photon Pt (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Background Photon Pt (endcap)',fontsize=20) #endcap only
plt.savefig(f'PFefficiency/' + sys.argv[1] +f'/phoPt_{sys.argv[1]}_bkgonly.png')
plt.close()




#################################################################################
txt.close()
print(f'\nAll done. Plots are saved in the folder : PFefficiency/{sys.argv[1]}\n')
