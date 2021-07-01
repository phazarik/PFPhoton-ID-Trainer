# PF-Photon-ID-Trainer
This is a machine learning code that utilises tensorflow and trains a sequential neural network to read into a csv file and distinguishes between real photons and fake photons. The goal is to distinguish between real photons and fake photons.

#### The CSV files :
The CSV files are stored in the directory ```../TrainingSamples/*```
Each row contains information about individual photons (like pT, eta, isPhotonMatching etc.)
These are flattened trees created from MINIAOD samples.
#### Running the program :
```
[] python PF-ID-trainer.py <modelname>.h5 <pdfname>.pdf 
```
This produces the model as an h5 file and the plots are stored in the pdf file.
The red dot in the ROC curve stands for the existing CMSSW flag.
