#############################################################################
#                 TRAINING A SEQUENTIAL NEURAL NETWORK                      #
#                                                                           #
# This code reads into CSV Files and file and trains the sequential NN      #
# it should be run as follows :                                             #
#                python PFPhoton-ID-Trainer.py <modelname>                  #
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

os.system("")
modelname = sys.argv[1]
os.system("mkdir -p output/" + modelname + "/extra_plots")
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('output/' + modelname + '/'+modelname+'.pdf')
#I added a text file which contains some information regarding the training process.
txt = open(f'output/' + modelname+ f'/max-min_{modelname}.txt', "w+")

#Do you want barrel or endcap?
isBarrel = False #True -> Barrel, False -> Endcap
#Do you want to debug?
isDebug = False #True -> nrows=1000

###################################################################################################################################################################

#READING DATA FILES :
print('Reading the input files')
#Columns: phoPt, phoEta, phoPhi, phoHoverE, phohadTowOverEmValid, photrkSumPtHollow, photrkSumPtSolid, phoecalRecHit, phohcalTower, phosigmaIetaIeta, phoSigmaIEtaIEtaFull5x5, phoSigmaIEtaIPhiFull5x5, phoEcalPFClusterIso, phoHcalPFClusterIso, phohasPixelSeed, phoR9Full5x5, isPhotonMatching, isPionMother, isPromptFinalState, isHardProcess, isPFPhoton (+ "sample" , "label" added later)

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

print('\nDefining the Signal File')

#signal1 = file1[(file1['isPhotonMatching'] == 1) and (file1['isPromptFinalState'] == 1) ]
signal2 = file2[(file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 1) ]
#signal3 = file3[(file3['isPhotonMatching'] == 1) and (file3['isPromptFinalState'] == 1)]

#######################################
# Defining the Background data-frames #
#######################################

print('\nDefining the Background File')

#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] == 0 ] 
#background2 = file2[file2['isPhotonMatching'] == 0 ]
#background3 = file3[file3['isPhotonMatching'] == 0 ]
background4 = file4[(file4['isPhotonMatching'] == 0) | ((file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 0)) ]

##################################################################
#Adding labels and sample column to distinguish varius samples

#signal1["sample"]=0
signal2["sample"]=1
#signal3["sample"]=2
#background1["sample"]=0
#background2["sample"]=1
#background3["sample"]=2
background4["sample"]=3

#signal1["label"]=1
signal2["label"]=1
#signal3["label"]=1
#background1["label"]=0
#background2["label"]=0
#background3["label"]=0
background4["label"]=0

################################################################
#Concatinating everything, and putting extra cuts:

Sig_alldf = pd.concat([signal2])

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

print("Removing Unnecessary Columns from Signal and Background Dataframes")
Sigdf = Sig_alldf.drop(['phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid', 'isPionMother'], axis=1)
Bkgdf = Bkg_alldf.drop(['phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid', 'isPionMother'], axis=1)
print('\nData reading succesful !\nBEGINNING THE TRAINING PROCESS\n')

##########################################################

#Final data frame creaton :    
data = pd.concat([Sigdf,Bkgdf])

print('dataframes are succesfully created.')
print('\nTotal Photons :')
print(data.shape)
print(data.head)

#data["weight"]=1
#Splitting:
data_train, data_test = train_test_split(data, test_size=0.5, stratify=data["label"])

n_signal_train = len(data_train.query('label == 1'))
n_signal_test = len(data_test.query('label == 1'))
n_bkg_train = len(data_train.query('label == 0'))
n_bkg_test = len(data_test.query('label == 0'))

print(f'signal photons (train) : {n_signal_train}')
print(f'signal photons (test) : {n_signal_test}')
print(f'bkg photons (train) : {n_bkg_train}')
print(f'bkg photons (test) : {n_bkg_test}')

txt.write("Number of Photons :\n\n")
txt.write(f'Total number of Signal Photons : {n_signal_train + n_signal_test}\n')
txt.write(f'Total number of Background Photons : {n_bkg_train + n_bkg_test}\n\n')
txt.write(f'No. of signal photons used for training : {n_signal_train}\n')
txt.write(f'No. of signal photons used for testing : {n_signal_test}\n')
txt.write(f'No.of bkg photons used for training : {n_bkg_train}\n')
txt.write(f'No. of bkg photons used for testing : {n_bkg_test}\n')


#weight :
#data_train.loc[data_train['label'] == 1, "weight"] =1/len(data_train.loc[data_train['label'] == 1])
#data_test.loc[data_test['label'] == 1, "weight"] =1/len(data_test.loc[data_test['label'] == 1])
#data_train.loc[data_train['label'] == 0, "weight"] =1/len(data_train.loc[data_train['label'] == 0])
#data_test.loc[data_test['label'] == 0, "weight"] =1/len(data_test.loc[data_test['label'] == 0])

#Splitting the label column as y and input variables as X
X_train= data_train[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values
y_train= data_train["label"].values #Takes only the label column (1=signal, 0=background)
#w_train= data_train["weight"].values
#print(f'Shapes of X, y are {X.shape} , {y.shape}')

X_test= data_test[['phoHoverE', 'photrkSumPtHollow', 'phoecalRecHit','phosigmaIetaIeta','phoSigmaIEtaIEtaFull5x5','phoSigmaIEtaIPhiFull5x5','phoEcalPFClusterIso','phoHcalPFClusterIso','phohasPixelSeed','phoR9Full5x5','phohcalTower']].values
y_test= data_test["label"].values #Takes only the label column (1=signal, 0=background)
#w_test= data_test["weight"].values

#print(data_train.columns)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

########################################################################
# NORMALISATION :
#Normalising X_train:
maxValues = X_train.max(axis=0) #column
minValues = X_train.min(axis=0)
print("\n### SAVE THE FOLLOWING VALUES ###")
print("Max values")
print(maxValues)
print("Min values")
print(minValues)
print('#################################\n')
MaxMinusMin = maxValues - minValues
normedX_train = 2*((X_train - minValues)/(MaxMinusMin)) -1.0
X_train = normedX_train
#Normalising X_test using the same Max and Min values as before:
normedX_test = 2*((X_test - minValues)/(MaxMinusMin)) -1.0
X_test = normedX_test
print("The train and test  data has been normalised.")
#print(type(maxValues))

#We need these max and min values for testing later. We can keep them in a text file.

txt.write(f'\nNormalisation Parameters :\n')
txt.write(f'\nmaxValues =\n')
txt.write('[')
for entries in maxValues :
    txt.write(str(entries)+', ')
txt.write(']')
txt.write(f'\nminValues =\n')
txt.write('[')
for entries in minValues :
    txt.write(str(entries)+', ')
txt.write(']')
#txt.close()
#######################################################################


n_features = X_train.shape[1]
print(f'\nNormalisation done.\nThe number of input variables is {n_features}\nThe training has begun.')


########################################################
#                The neural network :                  #
########################################################

#ACTIVATE ONLY ONE MODEL AT A TIME :

#model 1 :
#model = Sequential()
#model.add(Dense(n_features, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
#model.add(Dense(1, activation='sigmoid'))

#model2:
#model = Sequential()
#model.add(Dense(64, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
#model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(1, activation='sigmoid'))

#model 3 :
model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model :
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)

#Training the model :
history = model.fit(X_train,y_train,epochs=50,batch_size=1024,validation_data=(X_test,y_test),verbose=0)
#history = model.fit(X_train,y_train,epochs=50,batch_size=1024,validation_data=(X_test,y_test, w_test),verbose=0, sample_weight=w_train)

#Saving the output :
print('The NN architecture is')
model.summary()
model.save('output/'+ modelname +f'/{modelname}.h5')

print('\nTRAINING SUCESS!\n')
print('Plotting has begun.')


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
plt.savefig('output/'+modelname+'/loss_v_epoch.png')
plt.savefig(pp, format='pdf')

#plt.yscale('log')
#plt.title('Loss plot (log)', fontsize=20)
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
print('\nTest:')
print(f'shape of fpr = {fpr.shape}')
print(f'shape of tpr = {tpr.shape}')
print(f'shape of auc = {auc_score.shape}\n')
print(f'Shape of data_train and data_test = {data_train.shape} and {data_test.shape}')
print(f'Shape of the whole dataframe = {data.shape}')

##############################################################################
# Now we plot the NN output
#mybins = np.arange(0,1.05,0.05)
mybins = np.arange(0, 1.02, 0.02)

# First we make histograms to plot the testing data as points with errors
testsig = plt.hist(v_df[v_df['test_truth']==1]['test_prob'],bins=mybins, density=False)
testsige = np.sqrt(testsig[0])
testbkg = plt.hist(v_df[v_df['test_truth']==0]['test_prob'],bins=mybins, density=False)
testbkge = np.sqrt(testbkg[0])

#################################################################################################
#We store the NNScore in a new column :                                                         #
print('\nAdding the NN score to the test dataframe.')                                           #
print(data_test.shape)                                                                          #
data_test.loc[data_test['label'] == 1, "NNScore"] =  v_df[v_df['test_truth']==1]['test_prob']   #
data_test.loc[data_test['label'] == 0, "NNScore"] =  v_df[v_df['test_truth']==0]['test_prob']   #
print(f'NN score added. The dataframe looks like - ')                                           #
print(data_test.shape)                                                                          #
#################################################################################################

#NN-score plot
plt.figure(figsize=(8,6))
plt.errorbar(testsig[1][1:]-0.01, testsig[0], yerr=testsige, fmt='.', color="xkcd:green",label="Signal test", markersize='10')
plt.errorbar(testbkg[1][1:]-0.01, testbkg[0], yerr=testbkge, fmt='.', color="xkcd:denim",label="Background test", markersize='10')
plt.hist(t_df[t_df['train_truth']==1]['train_prob'],bins=mybins, histtype='step', label="Signal train", linewidth=3, color='xkcd:greenish',density=False,log=False)
plt.hist(t_df[t_df['train_truth']==0]['train_prob'],bins=mybins, histtype='step', label="Background train", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='best')
plt.xlabel('Score',fontsize=20)
plt.ylabel('Entries',fontsize=15)

if isBarrel == True :
    plt.title(f'NN Output (barrel photons)',fontsize=20) #barrel only
else :
    plt.title(f'NN Output (endcap photons)',fontsize=20) #endcap only

plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'output/'+ modelname + f'/NNscore_{modelname}.png')
plt.savefig(pp, format='pdf')
plt.yscale("log")
plt.savefig(f'output/'+ modelname + f'/NNscore_{modelname}_log.png')
plt.savefig(pp, format='pdf')
plt.close()

###############################################################################
#ROC plot
tpr=tpr*100
fnr=(1-fpr)*100
tpr1=tpr1*100
fnr1=(1-fpr1)*100
plt.figure(figsize=(8,8))
plt.plot(tpr,fnr,color='xkcd:denim blue', label='Training ROC (AUC = %0.4f)' % auc_score1)
plt.plot(tpr1,fnr1,color='xkcd:sky blue', label='Testing ROC (AUC = %0.4f)' % auc_score)

backgroundpass=len(data.query('(isPFPhoton == 1) & (label == 0)'))
backgroundrej =len(data.query('(isPFPhoton == 0) & (label == 0)'))
signalpass    =len(data.query('(isPFPhoton == 1) & (label == 1)'))
signalrej     =len(data.query('(isPFPhoton == 0) & (label == 1)'))
print(f'\nPhoton pass/fail info (PF-ID) :')
print(f'No of background photons = {len(Bkgdf)}')
print(f'No of background photons passed = {backgroundpass}')
print(f'No of background photons failed = {backgroundrej}')
print(f'No of signal photons = {len(Sigdf)}')
print(f'No of signal photons passed = {signalpass}')
print(f'No of signal photons failed = {signalrej}')
backgroundrej=( backgroundrej/(backgroundpass+backgroundrej) )*100
signaleff=( signalpass/(signalpass+signalrej) )*100
print(f'Background rejection = {backgroundrej}')
print(f'1 - background rejection (False positive rate) = {1-backgroundrej}')
print(f'signal efficiency (True Positive Rate) = {signaleff}')
plt.plot([signaleff], [backgroundrej], marker='o', color="red", markersize=6, label='CMSSW flag')

plt.legend(loc='lower right')
#plt.title(f'ROC Curve',fontsize=20)

if isBarrel == True :
    plt.title(f'ROC Curve (barrel photons)',fontsize=20) #barrel photons
else :
    plt.title(f'ROC Curve (endcap photons)',fontsize=20) #endcap photons

plt.xlabel('Signal Efficiency',fontsize=20)
plt.ylabel('Background Rejection',fontsize=20)
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig(f'output/'+modelname+ f'/ROC_{modelname}.png')
plt.savefig(pp, format='pdf')
plt.close()

###########################################################
#Done. Closing the opened files :
txt.close()
pp.close()
print(f'\nAll done. Model is saved as {modelname}\n')
