# PF-Photon-ID-Trainer
This is a collection of machine learning codes which utilises tensorflow and trains/evaluates a sequential neural network with a goal to distinguish between real photons and fake photons.

#### The CSV files :
The CSV files are stored in the directory ```../TrainingSamples```
Each row contains information about individual photons (like pT, eta, isPhotonMatching etc.)
These are flattened trees created from MINIAOD samples.
The ntuplizer and the root-to-csv converter are stored in the folder : ```CSV_maker```
#### Running the program :
##### Weight Calculation :
First calculate the weights associated with the photons in the different pT bins by running -
```
[] python weight_calculator.py <barrel/endcap> 
```
This produces some pT plots of the signal and background photons for the different samples (before and after applying the weights) inside a new folder : ```pT_reweigh```. It also generates a text file '*weight_<barrel/endcap>.txt*'. Copy and paste these weights into the trainer code.

##### Training the models:
After copying the contents from the *weight_<barrel/endcap>.txt file*, make sure to turn the weight on/off in the training. (In the *history=model.fit()* function). Then run the following.
```
[] python PFPhoton-ID-Trainer.py <modelname> <barrel/endcap>
```
This will create some files and plots in a new folder ```models/<modelname>```. The *scaler.txt* file will be automatically read by the Evaluation script in the next step. It will also pick up the *model.h5* file. The pb files are designed to be used in the CMSSW framework.

##### Evaluation and Finding Working Points:
After the training is done. The following evaluation script should be run to find some working points for the NN. It can be used to compare the performance of the neural network with the CMSSW flag.
```
[] python PFPhoton-ID-Evaluation.py <modelname> <barrel/endcap>
```
This code will produce an ROC with different working points in a new folder ```evaluated/<modelname>```. You can decide the cut on the NN score in this step, and apply it while making the efficiency plots.

##### Efficiency calculation in pT bins:
The following code calculates and plots the signal efficiency and background efficiency (fake rate) in different pT bins. One output plot is for the efficiencies of the NN at a specific cut, the second one is for efficiencies of the CMSSW flag. You can choose the value of the cut to be anything between 0 and 1, after looking at the evaluation ROC from the previous step.
```
[] python PFPhoton-ID-Efficiency.py <modelname> <barrel/endcap> <nn_cut>
```
This line will produce efficiency plots in a new folder : ```efficiency/<modelname>_<nn_cut>```



