import uproot
import glob
import pandas as pd

file_list = glob.glob("QCD_15to3000/egmNtuple_11*")    # This will create a list of all your files in the directory
Tree = "egmNtuplizer/EventTree"    # Tree location in the root file
branches=["phoPt", "phoEta", "phoPhi", "phoHoverE", "phohadTowOverEmValid", "photrkSumPtHollow", "photrkSumPtSolid", "phoecalRecHit", "phohcalTower", "phosigmaIetaIeta", "phoSigmaIEtaIEtaFull5x5","phoSigmaIEtaIPhiFull5x5", "phoEcalPFClusterIso", "phoHcalPFClusterIso", "phohasPixelSeed", "phoR9Full5x5","phohcalTower", "phohadOverEmCone", "isPhotonMatching", "isPionMother", "isPromptFinalState", "isHardProcess", "isPFPhoton"]    # Branches you want to select (Skimming)
cut="phoPt>10"    # Cuts (Trimming)
nameofdf="df_QCD2.csv.gzip"    # Name of final file

#----------------------------------------
df=pd.DataFrame()
for file in file_list:
    dfa=uproot.open(file)[Tree].pandas.df(branches=branches,flatten=True).reset_index(drop=True)
    dfa.query(cut,inplace=True)
    df=pd.concat([df,dfa])
df.to_csv(nameofdf,compression='gzip')
#----------------------------------------
