#############################################################################
#                 TRAINING A SEQUENTIAL NEURAL NETWORK                      #
#                                                                           #
# This code reads into CSV Files and file and trains the sequential NN.     #
# It should be run as follows :                                             #
#                                                                           #
#         python PFPhoton-ID-Trainer.py <modelname> <barrel/endcap>         #
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
os.system("mkdir -p models/" + modelname + "/extra_plots")
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('models/' + modelname + '/'+modelname+'.pdf')
#I added a text file which contains some information regarding the training process.
txt = open(f'models/' + modelname+ f'/info_{modelname}.txt', "w+")
scaler =  open(f'models/' + modelname+ f'/scaler_{modelname}.txt', "w+")


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
#new : phohadOverEmCone
#old : phoHoverE
batch_size_ = 1024
epochs_=100

###################################################################################################################################################################

#READING DATA FILES :
print('\nReading the input files.')

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
    
#####################################
#  Defining the Signal dataframes   #
#####################################

print('1. Defining Signal.')
#signal1 = file1[(file1['isPhotonMatching'] == 1) and (file1['isPromptFinalState'] == 1) ]
signal2 = file2[(file2['isPhotonMatching'] == 1) & (file2['isPromptFinalState'] == 1) ]
#signal3 = file3[(file3['isPhotonMatching'] == 1) and (file3['isPromptFinalState'] == 1)]
print('Done.')

#######################################
# Defining the Background data-frames #
#######################################

print('2. Defining Background.')
#### Adding conditions to the background file :
#background1 = file1[file1['isPhotonMatching'] == 0 ] 
#background2 = file2[file2['isPhotonMatching'] == 0 ]
#background3 = file3[file3['isPhotonMatching'] == 0 ]
background4 = file4[(file4['isPhotonMatching'] == 0) | ((file4['isPhotonMatching'] == 1) & (file4['isPromptFinalState'] == 0)) ]
background5 = file5[(file5['isPhotonMatching'] == 0) | ((file5['isPhotonMatching'] == 1) & (file5['isPromptFinalState'] == 0)) ]
print('Done.')

##################################################################
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

################################################################
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

print("Removing Unnecessary Columns from Signal and Background Dataframes.")
Sigdf = Sig_alldf.drop(['phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid', 'isPionMother'], axis=1)
Bkgdf = Bkg_alldf.drop(['phoPhi','isPhotonMatching','phohadTowOverEmValid', 'photrkSumPtSolid', 'isPionMother'], axis=1)

##########################################################

#Final data frame creaton :    
data = pd.concat([Sigdf,Bkgdf])

print('\nData reading succesful!')
print('Dataframes are succesfully created.')

##################################################
#        Adding weights to the training          #
##################################################

# pT bins :
# For signal photons :     0-20, 20-50 and 50+ : 3bins x 2samples = 6 bins 
# For background photons : 0-30, 30-60 and 60+ : 3bins x 2samples = 6 bins
# 12 bins in total.
# These values are calculated using a separate script.

if isBarrel == True:
    weight11_QCD = 52.166666666666664
    weight12_QCD = 337.1200980392157
    weight13_QCD = 19.604195804195804
    weight01_QCD = 3.316458154179034
    weight02_QCD = 0.3721821907846896
    weight03_QCD = 0.008004726037512924
    weight11_Tau = 0
    weight12_Tau = 0
    weight13_Tau = 0
    weight01_Tau = 6.719593075539568
    weight02_Tau = 0.33872811503519645
    weight03_Tau = 0.005864023888864846
else:
    weight11_QCD = 35.29959514170041
    weight12_QCD = 342.70355731225294
    weight13_QCD = 37.609375
    weight01_QCD = 4.41288971878006
    weight02_QCD = 0.8619804204782539
    weight03_QCD = 0.012523822488429077
    weight11_Tau = 0
    weight12_Tau = 0
    weight13_Tau = 0
    weight01_Tau = 4.137786502546689
    weight02_Tau = 0.4072024260803639
    weight03_Tau = 0.01759979594439485

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

#########################################
#Splitting:
data_train, data_test = train_test_split(data, test_size=0.5, stratify=data["label"])

n_signal_train = len(data_train.query('label == 1'))
n_signal_test = len(data_test.query('label == 1'))
n_bkg_train = len(data_train.query('label == 0'))
n_bkg_test = len(data_test.query('label == 0'))


print('\nSTATISTICS:')
print(f'Total number of Signal Photons : {n_signal_train + n_signal_test} = {n_signal_train} (train) + {n_signal_test} (test)')
print(f'Total number of Background Photons : {n_bkg_train + n_bkg_test} = {n_bkg_train} (train) + {n_bkg_test} (test)')

#Splitting the label column as y and input variables as X
X_train= data_train[train_var].values
y_train= data_train["label"].values #Takes only the label column (1=signal, 0=background)
w_train= data_train["weight"].values
#print(f'Shapes of X, y are {X.shape} , {y.shape}')

X_test= data_test[train_var].values
y_test= data_test["label"].values #Takes only the label column (1=signal, 0=background)
w_test= data_test["weight"].values

#removed : 'phoEcalPFClusterIso','phoHcalPFClusterIso',

#print(data_train.columns)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

########################################################################
# NORMALISATION :
#Normalising X_train:
maxValues = X_train.max(axis=0) #column
minValues = X_train.min(axis=0)
print("\nNORMALISATION PARAMETERS :")
print(f"Max values = {maxValues}")
print(f"Min values = {minValues}")

MaxMinusMin = maxValues - minValues
normedX_train = 2*((X_train - minValues)/(MaxMinusMin)) -1.0
X_train = normedX_train
#Normalising X_test using the same Max and Min values as before:
normedX_test = 2*((X_test - minValues)/(MaxMinusMin)) -1.0
X_test = normedX_test
print("The train and test  data has been normalised.")
#We need these max and min values for testing later. We can keep them in a text file.

#Writing the scalar file:
i=0
while i < len(varnames) :
    scaler.write(f"{varnames[i]} minmax {minValues[i]} {maxValues[i]}\n")
    i = i+1


n_features = X_train.shape[1]
print(f'\nThe number of input variables is {n_features}\nLoading the Neural Network.\n')


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
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_dim=n_features, name='FirstLayer'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid', name='FinalLayer'))

#Compiling the model :
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)

#Training the model :
#history = model.fit(X_train,y_train,epochs=epochs_,batch_size=batch_size_,validation_data=(X_test,y_test),verbose=0) #Without weight
history = model.fit(X_train,y_train,epochs=epochs_,batch_size=batch_size_,validation_data=(X_test,y_test, w_test),verbose=0, sample_weight=w_train) #With weight

#Saving the output :
print('The NN architecture is')
model.summary()
model.save('models/'+ modelname +f'/{modelname}.h5')
#model.save('output/'+ modelname +f'/{modelname}_withoutcmsml')

import cmsml as cmsml
cmsml.tensorflow.save_graph(f"models/{modelname}/{modelname}.pb.txt", model, variables_to_constants=True)
cmsml.tensorflow.save_graph(f"models/{modelname}/{modelname}.pb", model, variables_to_constants=True)


print('\nTRAINING SUCESS!')
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
plt.savefig('models/'+modelname+'/loss_v_epoch.png')
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
plt.savefig(f'models/'+ modelname + f'/NNscore_{modelname}.png')
plt.savefig(pp, format='pdf')
plt.yscale("log")
plt.savefig(f'models/'+ modelname + f'/NNscore_{modelname}_log.png')
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
print(f'Background rejection = {backgroundrej:.0f}%')
print(f'100 - background rejection (False positive rate) = {100-backgroundrej:.0f}%')
print(f'signal efficiency (True Positive Rate) = {signaleff:.0f}%')
plt.plot(signaleff, backgroundrej, marker='o', color="red", markersize=6, label=f'CMSSW flag ({signaleff:.0f},{backgroundrej:.0f})')

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
plt.savefig(f'models/'+modelname+ f'/ROC_{modelname}.png')
plt.savefig(pp, format='pdf')
plt.close()

###########################################################
#Writing the text file :
#Model Information:
txt.write("MODEL INFORMATION:\n")
txt.write(f"Name = {modelname}\n")
txt.write("Sequential : 128/64/64/32/16/1\n")
txt.write(f"batch size = {batch_size_}, epochs={epochs_}\n")

#list of variables:
i=0
txt.write("\nLIST OF TRAINING VARIABLES :\n")
while i<(len(varnames)):
    txt.write(f"{i+1}. {varnames[i]}\n")
    i=i+1

#Normalisation Paremeters:
txt.write(f'\nNORMALISATION PARAMETERS (min-max) :')
txt.write(f'\nmaxValues =')
txt.write('[')
for entries in maxValues :
    txt.write(str(entries)+', ')
txt.write(']')
txt.write(f'\nminValues =')
txt.write('[')
for entries in minValues :
    txt.write(str(entries)+', ')
txt.write(']')

#Statistics:
txt.write("\n\nNUMBER OF PHOTONS :")
txt.write(f'\nTotal number of Signal Photons : {n_signal_train + n_signal_test} = {n_signal_train} (train) + {n_signal_test} (test)')
txt.write(f'\nTotal number of Background Photons : {n_bkg_train + n_bkg_test} = {n_bkg_train} (train) + {n_bkg_test} (test)')
QCD_contribution = len(data.query('(label==0) & (sample==3)'))
Tau_contribution = len(data.query('(label==0) & (sample==4)'))
QCD_contribution_frac = (QCD_contribution*100) / (QCD_contribution+Tau_contribution)
Tau_contribution_frac = (Tau_contribution*100) / (QCD_contribution+Tau_contribution)
txt.write(f'\nContribution from QCD file = {QCD_contribution} ({QCD_contribution_frac:.0f}%)')
txt.write(f'\nContribution from TauGun file = {Tau_contribution} ({Tau_contribution_frac:.0f}%)')

#Result:
txt.write(f'\n\n RESULT :')
txt.write(f'\nSignal Efficiency (PF) = {signaleff:.0f}%')
txt.write(f'\nBackground Rejection (PF) = {backgroundrej:.0f}%')


#Done. Closing the opened files :
txt.close()
scaler.close()
pp.close()
print(f'\nAll done. Model is saved as models/{modelname}\n')
