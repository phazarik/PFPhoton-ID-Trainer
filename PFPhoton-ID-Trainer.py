#############################################################################
#                 TRAINING A SEQUENTIAL NEURAL NETWORK                      #
#                                                                           #
# This code reads into CSV Files and file and trains the sequential NN      #
# it should be run as follows :                                             #
#     Training_PFID_alternative.py model_name.h5 output_name.pdf            #
# NOTE : Save the max and minvalues from the terminal output for step2      #
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

modelname = sys.argv[1]
outputname = sys.argv[2]

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

print('Reading the input files')
#Reading the data :
#signal1 = pd.read_csv('../TrainingSamples/GJet_20to40/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
signal2 = pd.read_csv('../TrainingSamples/GJet_20toinf/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
#signal3 = pd.read_csv('../TrainingSamples/GJet_40toinf/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

background1=pd.read_csv('../TrainingSamples/QCD/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
background2=pd.read_csv('../TrainingSamples/TauGun/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

#print(signal1.head)

# I need to add the condition that, for signal, isPhotonMatching =1
# and for background, isPhotonMatching = 0
# additonal cuts can also be added here, for example, the PF-ID flags

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

########################
#   Signal dataframe   #
########################

print('\nReading into the Signal File')

Sig_alldf = pd.concat([signal2])

#### Adding cuts to the signal file :
Sig_alldf = Sig_alldf[Sig_alldf['isPhotonMatching'] == 1]#keep the rows that contain gen-matching photons
#Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) < 1.442] #barrel
#Sig_alldf = Sig_alldf[abs(Sig_alldf['phoEta']) > 1.566] #endcap

#print('Variables in the data file :')
#print(list(Sig_alldf.columns))
        
# Adding the isPFPhoton flags as the last column in the signal dataframe.
# If all of they above condition are true, only then their product is true,
# which means the photon is a  PF-photon
print('Adding the PF flags to the signal file')
Sig_alldf['isPFphoton'] = Sig_alldf.apply(lambda row: flag1(row.phoPt)*flag2(row.phoHoverE)*flag3(row.photrkSumPtHollow, row.phoecalRecHit, row.phohcalTower)*flag4(row.phohadTowOverEmValid, row.photrkSumPtSolid, row.phoPt)*flag5(row.phoEta, row.phosigmaIetaIeta), axis=1)
#print(Sig_alldf.head)
print('PF Flags are added succesfully!\nNow dropping the unnecessary columns.')
#removing the unnecessary columns :
Sigdf = Sig_alldf.drop(['phoPt','phoEta','phoPhi','isPhotonMatching','isPromptFinalState','phohadTowOverEmValid'], axis=1)
print('Unnecessary columns dropped from the signal File.\nThe dataframe has the following structure :')
print(Sigdf.shape)
print(list(Sigdf.columns))
print(Sigdf.head)


#########################
# Background data-frame #
#########################

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
print('\nData reading succesful !\nBEGINNING THE TRAINING PROCESS\n')


##########################################################

#Final data frame creaton :
Sigdf["label"]=1           #This creates an additional column "label"
Bkgdf["label"]=0
    
data = pd.concat([Sigdf,Bkgdf])
print('dataframes are succesfully created.')

#Splitting the label column as y and input variables as X
X, y = data.values[:,:-1], data.values[:,-1]
print(f'Shapes of data, X, y are {data.shape}, {X.shape} , {y.shape}')


########################################################
# Normalize the input variables to go from -1 to 1     #
# using  normedX = 2(X - min)/(max - min) - 1.0        #
maxValues = X.max(axis=0)                              #
minValues = X.min(axis=0)                              #
print("\n### SAVE THE FOLLOWING VALUES ###")           #
print("Max values")                                    #
print(maxValues)                                       #
print("Min values")                                    #
print(minValues)                                       #
print('#################################\n')           #
MaxMinusMin = X.max(axis=0) - X.min(axis=0)            #
normedX = 2*((X-X.min(axis=0))/(MaxMinusMin)) -1.0     #
X = normedX                                            #
########################################################
# This normalisation procedure is not useful when you have variables containing
# all zeros or all values close to zero. We have to be careful with this. 
# Currently, the variable 'phohadTowOverEmValid' has been removed because of this.

#Splitting the data into a training and a testing part :
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')



########################################################
#                The neural network :                  #
########################################################

model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model :
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the model :
history = model.fit(X_train,y_train,epochs=100,batch_size=1024,validation_data=(X_test,y_test),verbose=0)

#Saving the output :
print('The NN architecture is')
model.summary()
model.save(modelname)

print('\nTRAINING SUCESS!\n')

###########################################################
#Fancy plotting to see how the training performed :
#Figure1 : accuracy vs epoch
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.4, 1])
plt.legend(loc='upper left')
#plt.savefig('acc_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()

#Figure2 : loss vs epoch
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
plt.title('Loss plot', fontsize=20)
#plt.savefig('loss_v_epoch.png')
plt.savefig(pp, format='pdf')

plt.yscale('log')
plt.title('Loss plot (log)', fontsize=20)
plt.close()




############################################################
#             Plotting the neural network score :          #
############################################################

# Now that those are done, we will plot the score
# We plot the score separately for signal and background
# and separately for testing and training datasets
# Thus there are 4 curves plotted.

#Setup some new dataframes  t_df is testing, v_df is training (or validation)
t_df = pd.DataFrame()
v_df = pd.DataFrame()
t_df['train_truth'] = y_train
t_df['train_prob'] = 0
v_df['test_truth'] = y_test
v_df['test_prob'] = 0

# Now we evaluate the model on the test and train data by calling the
# predict function

val_pred_proba = model.predict(X_test)
train_pred_proba = model.predict(X_train)
t_df['train_prob'] = train_pred_proba
v_df['test_prob'] = val_pred_proba

# Okay so now we have the two dataframes ready.
# t_df has two columns for training data  (train_truth and train_prob)
# v_df has two columns for testing data  (train_truth and train_prob)

# Now we get the ROC curve, first for testing
fpr, tpr, _ = roc_curve(y_test,val_pred_proba)
auc_score = auc(tpr,1-fpr)
# Now the ROC curve for training data
fpr1, tpr1, _ = roc_curve(y_train,train_pred_proba)
auc_score1 = auc(tpr1,1-fpr1)


# Now we plot the NN output
#mybins = np.arange(0,1.05,0.05)
mybins = np.arange(0, 1.02, 0.02)

# First we make histograms to plot the testing data as points with errors
testsig = plt.hist(v_df[v_df['test_truth']==1]['test_prob'],bins=mybins)
testsige = np.sqrt(testsig[0])
testbkg = plt.hist(v_df[v_df['test_truth']==0]['test_prob'],bins=mybins)
testbkge = np.sqrt(testbkg[0])

#NN-score plot
plt.figure(figsize=(8,6))
plt.errorbar(testsig[1][1:]-0.025, testsig[0], yerr=testsige, fmt='.', color="xkcd:green",label="Signal test", markersize='10')
plt.errorbar(testbkg[1][1:]-0.025, testbkg[0], yerr=testbkge, fmt='.', color="xkcd:denim",label="Background test", markersize='10')
plt.hist(t_df[t_df['train_truth']==1]['train_prob'],bins=mybins, histtype='step', label="Signal train", linewidth=3, color='xkcd:greenish',density=False,log=False)
plt.hist(t_df[t_df['train_truth']==0]['train_prob'],bins=mybins, histtype='step', label="Background train", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='upper center')
plt.xlabel('Score',fontsize=20)
plt.ylabel('Events',fontsize=15)
plt.title(f'NN Output',fontsize=20)
#plt.title(f'NN Output (barrel photons)',fontsize=20) #barrel only
#plt.title(f'NN Output (endcap photons)',fontsize=20) #endcap only
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'NNscore_{outputname}.png')
plt.savefig(pp, format='pdf')
plt.yscale("log")
plt.savefig(f'NNscore_{outputname}_log.png')
plt.savefig(pp, format='pdf')
plt.close()

#ROC plot
tpr=tpr*100
fnr=(1-fpr)*100
tpr1=tpr1*100
fnr1=(1-fpr1)*100
plt.figure(figsize=(8,8))
plt.plot(tpr,fnr,color='xkcd:denim blue', label='Training ROC (AUC = %0.4f)' % auc_score1)
plt.plot(tpr1,fnr1,color='xkcd:sky blue', label='Testing ROC (AUC = %0.4f)' % auc_score)
'''
backgroundpass=data.loc[(data['isPFphoton'] == 1) & (data['label'] == 0)].sum()
backgroundrej=data.loc[(data['isPFphoton'] == 0) & (data['label'] == 0)].sum()
signalpass=data.loc[(data['isPFphoton'] == 1) & (data['label'] == 1)].sum()
signalrej=data.loc[(data['isPFphoton'] == 0) & (data['label'] == 1)].sum()
'''
backgroundpass=len(data.query('(isPFphoton == 1) & (label == 0)'))
backgroundrej =len(data.query('(isPFphoton == 0) & (label == 0)'))
signalpass    =len(data.query('(isPFphoton == 1) & (label == 1)'))
signalrej     =len(data.query('(isPFphoton == 0) & (label == 1)'))
print(f'\nPhoton pass/fail info :')
print(f'No of background photons passed = {backgroundpass}')
print(f'No of background photons failed = {backgroundrej}')
print(f'No of signal photons passed = {signalpass}')
print(f'No of signal photons failed = {signalrej}')
backgroundrej=( backgroundrej/(backgroundpass+backgroundrej) )*100
signaleff=( signalpass/(signalpass+signalrej) )*100
print(f'Background rejection = {backgroundrej}')
print(f'1 - background rejection (False positive rate) = {1-backgroundrej}')
print(f'signal efficiency (True Positive Rate) = {signaleff}')
plt.plot([signaleff], [backgroundrej], marker='o', color="red", markersize=6, label='CMSSW flag')

plt.legend(loc='lower right')
plt.title(f'ROC Curve',fontsize=20)
#plt.title(f'ROC Curve (barrel photons)',fontsize=20) #barrel photons
#plt.title(f'ROC Curve (endcap photons)',fontsize=20) #endcap photons
plt.xlabel('Signal Efficiency',fontsize=20)
plt.ylabel('Background Rejection',fontsize=20)
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig(f'ROC_{outputname}.png')
plt.savefig(pp, format='pdf')
plt.close()


pp.close()
print(f'\nAll done. Model is saved as {modelname}\n')
