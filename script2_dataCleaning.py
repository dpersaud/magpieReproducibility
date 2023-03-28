#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Monday, March 17th, 2023
Author: Daniel Persaud
Description: This script cleans the featurized dataset and produces 2 csv's which will be useful for the next script:
 - 

and outputs a subset of the data which:
- takes the lowest-energy compound for each unique composition
- has a value for bandgap
- has a bandgap energy between 0eV/atom and 1.5eV/atom
- is a halogen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# IMPORT THE FEATUREIZED DATA------------------------------------------------------------------------------------------
# read in the featurized data from script 1 (magpieReproducibilityChallange_1_dataFeaturization.py)
dfOQMD_wFeatures_wElementFrac = pd.read_csv('data/oqmd_all_featurized.csv', low_memory=False)
# replace 'None' string values with np.nan
dfOQMD_wFeatures_wElementFrac.replace('None', np.nan, inplace=True)
# make a list of all the columns that need to be cast to floats
lstFloatCols_oqmd = dfOQMD_wFeatures_wElementFrac.columns.tolist()
lstFloatCols_oqmd.remove('strComposition')
lstFloatCols_oqmd.remove('composition')
lstFloatCols_oqmd.remove('ElementFraction Exceptions')
lstFloatCols_oqmd.remove('ElementProperty Exceptions')
# cast all the columns to floats
dfOQMD_wFeatures_wElementFrac[lstFloatCols_oqmd] = dfOQMD_wFeatures_wElementFrac[lstFloatCols_oqmd].astype(float)

# read in the featurized data from script 1 (magpieReproducibilityChallange_1_dataFeaturization.py)
dfBandgap_wFeatures_wElementFrac = pd.read_csv('data/bandgap_featurized.csv', low_memory=False)
# replace 'None' string values with np.nan
dfBandgap_wFeatures_wElementFrac.replace('None', np.nan, inplace=True)
# make a list of all the columns that need to be cast to floats
lstFloatCols_bandgap = dfBandgap_wFeatures_wElementFrac.columns.tolist()
lstFloatCols_bandgap.remove('strComposition')
lstFloatCols_bandgap.remove('composition')
lstFloatCols_bandgap.remove('ElementFraction Exceptions')
lstFloatCols_bandgap.remove('ElementProperty Exceptions')
# cast all the columns to floats
dfBandgap_wFeatures_wElementFrac[lstFloatCols_bandgap] = dfBandgap_wFeatures_wElementFrac[lstFloatCols_bandgap].astype(float)


# OUTPUT: 
#       dfOQMD_wFeatures_wElementFrac: pandas.DataFrame
#           contains the featurized data for the OQMD dataset with all columns of the correct type
#       dfBandgap_wFeatures_wElementFrac: pandas.DataFrame
#           contains the featurized data for the bandgap dataset with all columns of the correct type
#%%
# CLEAN THE FEATURIZED DATA TO REMOVE DUPLICATES-----------------------------------------------------------------------
# remove duplicate compositions by selecting the lowest energy compound for every duplicate
dfOQMD_wFeatures_wElementFrac_clean = dfOQMD_wFeatures_wElementFrac.loc[dfOQMD_wFeatures_wElementFrac.groupby('strComposition')['energy_pa'].idxmin()]
# save the cleaned data to a csv file
dfOQMD_wFeatures_wElementFrac_clean.to_csv('data/oqmd_all_featurized_clean.csv')

# remove rows that contains 
# %%
