#############################################################################
#                         READING INTO THE CSV FILES                        #
# This program reads into the CSV files and prints out their shape and head #
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


from matplotlib.backends.backend_pdf import PdfPages
#pp = PdfPages(outputname)

print('Reading the input files')
#Reading the data :
signal1 = pd.read_csv('../TrainingSamples/TauGun/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
print('\nTotal size of the TauGun dataframe is -')
print(signal1.shape)
signalpass = signal1[signal1['isPhotonMatching'] == 1]
signalfail = signal1[signal1['isPhotonMatching'] == 0]
print('The size of TauGun Real Photon dataframe is -')
print(signalpass.shape)
print('The size of TauGun Fake Photon dataframe is -')
print(signalfail.shape)


background1=pd.read_csv('../TrainingSamples/QCD/df.csv.gzip',compression='gzip', usecols=[1,2,3,4,5,6,7,8,9,10,11,12]) #nrows=1000 #add this, if you want to debug
print('\nTotal size of the QCD dataframe is -')
print(background1.shape)
backgroundpass = background1[background1['isPhotonMatching'] == 1]
backgroundfail = background1[background1['isPhotonMatching'] == 0]
print('The size of QCD Real Photon dataframe is -')
print(backgroundpass.shape)
print('The size of QCD Fake Photon dataframe is -')
print(backgroundfail.shape)

