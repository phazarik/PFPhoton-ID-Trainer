#############################################################################
#                               pT reweighing                               #
#                                                                           #
# This code reads into CSV Files of different samples, and reweigh the      #
# photons in pT bins.                                                       #
# It should be run as follows :                                             #
#                                                                           #
#                python weight_calculator <barrel/endcap>                   #
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
os.system("mkdir -p pT_reweigh/")
txt = open(f'pT_reweigh/weights_{sys.argv[1]}.txt', "w+")

##########################################################
#                    Settings:                           #
##########################################################
isNorm = True
#Do you want to debug?
isDebug = False #True -> nrows=1000

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
mycols = [1, 2, 3, 18, 19, 20, 21, 22]
#mycols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
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


#########################################
#     plotting before reweighing        #
#########################################
location = 'center right'
bins_= np.arange(0, 151, 1)

#plot 1 : pT spectrum (signal):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 1) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False)
plt.hist(data.query('(label == 1) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False)
plt.hist(data.query('(label == 1) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False)
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Signal Pt without weights (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Signal Pt without weight (endcap)',fontsize=20) #endcap only
plt.savefig(f'pT_reweigh/phoPt_beforeW_sig_{sys.argv[1]}.png')
plt.close()

#plot 2 : pT spectrum (background):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 0) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False)
plt.hist(data.query('(label == 0) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False)
plt.hist(data.query('(label == 0) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False)
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Background Pt without weight (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Background Pt without weight (endcap)',fontsize=20) #endcap only
plt.savefig(f'pT_reweigh/phoPt_beforeW_bkg_{sys.argv[1]}.png')
plt.close()


#########################################
#             pT rewighing              #
#########################################

#pT bins :
#For signal photons : 0-20, 20-50 and 50+
#For background photons : 0-30, 30-60, 60+

#pT weights :
#We are using the GJet sample as a reference, so weights of the GJet sample = 1

#We ajust the weights of QCD and TauGun in different pT bins.
#For now, these weights are calculated in a different script.

#For QCD :
#weight of QCD in a pT bin = no. of GJet photons in that bin/ no. of QCD photons in that bin
#Signal photons:
if len( data.query('(label == 1) & (sample==3) & (phoPt < 20)') ) == 0 :
    weight11_QCD = 0
else:
    weight11_QCD = len( data.query('(label == 1) & (sample==1) & (phoPt < 20)') ) / len( data.query('(label == 1) & (sample==3) & (phoPt < 20)') )

if len( data.query('(label == 1) & (sample==3) & (phoPt > 20) & (phoPt < 50)') ) == 0 :
    weight12_QCD = 0
else :
    weight12_QCD = len( data.query('(label == 1) & (sample==1) & (phoPt > 20) & (phoPt < 50)') ) / len( data.query('(label == 1) & (sample==3) & (phoPt > 20) & (phoPt < 50)') )

if len( data.query('(label == 1) & (sample==3) & (phoPt > 50)') ) ==0:
    weight13_QCD = 0
else :
    weight13_QCD = len( data.query('(label == 1) & (sample==1) & (phoPt > 50)') ) / len( data.query('(label == 1) & (sample==3) & (phoPt > 50)') )

#background photons:
if len( data.query('(label == 0) & (sample==3) & (phoPt < 30)') ) == 0:
    weight01_QCD = 0
else:
    weight01_QCD = len( data.query('(label == 0) & (sample==1) & (phoPt < 30)') ) / len( data.query('(label == 0) & (sample==3) & (phoPt < 30)') )

if len( data.query('(label == 0) & (sample==3) & (phoPt > 30) & (phoPt < 60)') ) == 0:
    weight02_QCD = 0
else:
    weight02_QCD = len( data.query('(label == 0) & (sample==1) & (phoPt > 30) & (phoPt < 60)') ) / len( data.query('(label == 0) & (sample==3) & (phoPt > 30) & (phoPt < 60)') )

if len( data.query('(label == 0) & (sample==3) & (phoPt > 60)') ) == 0:
    weight03_QCD = 0
else :
    weight03_QCD = len( data.query('(label == 0) & (sample==1) & (phoPt > 60)') ) / len( data.query('(label == 0) & (sample==3) & (phoPt > 60)') )

#For TauGun :
#Signal photons:
#There are no signal photons from TauGun, but we check for it anyway
if len( data.query('(label == 1) & (sample==4) & (phoPt < 20)') ) == 0:
    weight11_Tau = 0
else :
    weight11_Tau = len( data.query('(label == 1) & (sample==1) & (phoPt < 20)') ) / len( data.query('(label == 1) & (sample==4) & (phoPt < 20)') )

if len( data.query('(label == 1) & (sample==4) & (phoPt > 20) & (phoPt < 50)') ) == 0:
    weight12_Tau = 0
else :
    weight12_Tau = len( data.query('(label == 1) & (sample==1) & (phoPt > 20) & (phoPt < 50)') ) / len( data.query('(label == 1) & (sample==4) & (phoPt > 20) & (phoPt < 50)') )

if len( data.query('(label == 1) & (sample==4) & (phoPt > 50)') ) == 0:
    weight13_Tau = 0
else :
    weight13_Tau = len( data.query('(label == 1) & (sample==1) & (phoPt > 50)') ) / len( data.query('(label == 1) & (sample==4) & (phoPt > 50)') )

#background photons:
if len( data.query('(label == 0) & (sample==4) & (phoPt < 30)') ) == 0:
    weight01_Tau = 0
else:
    weight01_Tau = len( data.query('(label == 0) & (sample==1) & (phoPt < 30)') ) / len( data.query('(label == 0) & (sample==4) & (phoPt < 30)') )

if len( data.query('(label == 0) & (sample==4) & (phoPt > 30) & (phoPt < 60)') ) == 0:
    weight02_Tau = 0
else:
    weight02_Tau = len( data.query('(label == 0) & (sample==1) & (phoPt > 30) & (phoPt < 60)') ) / len( data.query('(label == 0) & (sample==4) & (phoPt > 30) & (phoPt < 60)') )

if len( data.query('(label == 0) & (sample==4) & (phoPt > 60)') ) == 0:
    weight03_Tau = 0
else:
    weight03_Tau = len( data.query('(label == 0) & (sample==1) & (phoPt > 60)') ) / len( data.query('(label == 0) & (sample==4) & (phoPt > 60)') )

print("\nThe weights are :")
print(f"weight11_QCD = {weight11_QCD}")
print(f"weight12_QCD = {weight12_QCD}")
print(f"weight13_QCD = {weight13_QCD}")
print(f"weight01_QCD = {weight01_QCD}")
print(f"weight02_QCD = {weight02_QCD}")
print(f"weight03_QCD = {weight03_QCD}")

print(f"weight11_Tau = {weight11_Tau}")
print(f"weight12_Tau = {weight12_Tau}")
print(f"weight13_Tau = {weight13_Tau}")
print(f"weight01_Tau = {weight01_Tau}")
print(f"weight02_Tau = {weight02_Tau}")
print(f"weight03_Tau = {weight03_Tau}")

txt.write(f"weight11_QCD = {weight11_QCD}\n")
txt.write(f"weight12_QCD = {weight12_QCD}\n")
txt.write(f"weight13_QCD = {weight13_QCD}\n")
txt.write(f"weight01_QCD = {weight01_QCD}\n")
txt.write(f"weight02_QCD = {weight02_QCD}\n")
txt.write(f"weight03_QCD = {weight03_QCD}\n")

txt.write(f"weight11_Tau = {weight11_Tau}\n")
txt.write(f"weight12_Tau = {weight12_Tau}\n")
txt.write(f"weight13_Tau = {weight13_Tau}\n")
txt.write(f"weight01_Tau = {weight01_Tau}\n")
txt.write(f"weight02_Tau = {weight02_Tau}\n")
txt.write(f"weight03_Tau = {weight03_Tau}\n")


#########################################
# Adding these weights to the dataframe :

data.loc[data['sample']==1, "weight"] = 1

data.loc[(data['sample']==3) & (data['label']==1) & (data['phoPt']<20), "weight"] = weight11_QCD
data.loc[(data['sample']==3) & (data['label']==1) & (data['phoPt']>20) & (data['phoPt']<50), "weight"] = weight12_QCD
data.loc[(data['sample']==3) & (data['label']==1) & (data['phoPt']>50), "weight"] = weight13_QCD

data.loc[(data['sample']==3) & (data['label']==0) & (data['phoPt']<30), "weight"] = weight01_QCD
data.loc[(data['sample']==3) & (data['label']==0) & (data['phoPt']>30) & (data['phoPt']<60), "weight"] = weight02_QCD
data.loc[(data['sample']==3) & (data['label']==0) & (data['phoPt']>60), "weight"] = weight03_QCD

data.loc[(data['sample']==4) & (data['label']==1) & (data['phoPt']<20), "weight"] = weight11_Tau
data.loc[(data['sample']==4) & (data['label']==1) & (data['phoPt']>20) & (data['phoPt']<50), "weight"] = weight12_Tau
data.loc[(data['sample']==4) & (data['label']==1) & (data['phoPt']>50), "weight"] = weight13_Tau

data.loc[(data['sample']==4) & (data['label']==0) & (data['phoPt']<30), "weight"] = weight01_Tau
data.loc[(data['sample']==4) & (data['label']==0) & (data['phoPt']>30) & (data['phoPt']<60), "weight"] = weight02_Tau
data.loc[(data['sample']==4) & (data['label']==0) & (data['phoPt']>60), "weight"] = weight03_Tau

print('test')
print(data.query('(sample==3)').head)
print('test done')

print(data.head)

#########################################
#      plotting after reweighing        #
#########################################
location = 'center right'
bins_= np.arange(0, 151, 1)


#plot 1 : pT spectrum (signal):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 1) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False, weights=data.query('(label == 1) & (sample ==3)')['weight'])
plt.hist(data.query('(label == 1) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False, weights=data.query('(label == 1) & (sample ==4)')['weight'])
plt.hist(data.query('(label == 1) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False, weights=data.query('(label == 1) & (sample ==1)')['weight'])
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Signal Pt with weights (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Signal Pt with weight (endcap)',fontsize=20) #endcap only
plt.savefig(f'pT_reweigh/phoPt_afterW_sig_{sys.argv[1]}.png')
plt.close()

#plot 2 : pT spectrum (background):
plt.figure(figsize=(8,6))
plt.hist(data.query('(label == 0) & (sample ==3)')['phoPt'], bins=bins_, histtype='step', label="QCD", linewidth=2, color='xkcd:denim',density=isNorm,log=False, weights=data.query('(label == 0) & (sample ==3)')['weight'])
plt.hist(data.query('(label == 0) & (sample ==4)')['phoPt'], bins=bins_, histtype='step', label="TauGun", linewidth=2, color='xkcd:red',density=isNorm,log=False, weights=data.query('(label == 0) & (sample ==4)')['weight'])
plt.hist(data.query('(label == 0) & (sample ==1)')['phoPt'], bins=bins_, histtype='step', label="GJet", linewidth=2, color='xkcd:green',density=isNorm,log=False, weights=data.query('(label == 0) & (sample ==1)')['weight'])
plt.legend(loc='best')
plt.xlabel('phoPt',fontsize=20)
plt.ylabel('Entries',fontsize=15)
if isBarrel == True :
    plt.title(f'Background Pt with weight (barrel)',fontsize=20) #barrel only
else :
    plt.title(f'Background Pt with weight (endcap)',fontsize=20) #endcap only
plt.savefig(f'pT_reweigh/phoPt_afterW_bkg_{sys.argv[1]}.png')
plt.close()
 


#################################################################################
txt.close()
print(f'\nAll done. Plots are saved in the folder plots\n')
