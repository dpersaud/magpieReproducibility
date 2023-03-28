#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Tuesday, March 28th, 2023

Author: Daniel Persaud

Description: This script cleans the featurized dataset and produces 2 csv's which will be used for the ML models:
 -  the first is OQMD with duplicate compositions removed by taking the lowest-energy compound for each unique 
    composition > this will be used for the 10-fold CV (originally used for model selection)
 -  the second is the lowest-energy compound for each unique halogen with a bandgap value between 0eV/atom and
    1.5eV/atom > this will be used to predict the the bandgap energies of the 5 compounds in table 5 of the original
    paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# IMPORT THE FEATUREIZED DATA------------------------------------------------------------------------------------------
# read in the featurized data from script 1 (script1_out_oqmd_all_featurized.py)
dfOQMD_wFeatures_wElementFrac = pd.read_csv('data/script1_out_oqmd_all_featurized.csv', low_memory=False)
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

# read in the featurized data from script 1 (script1_out_bandgap_featurized.py)
dfBandgap_wFeatures_wElementFrac = pd.read_csv('data/script1_out_bandgap_featurized.csv', low_memory=False)
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

# read in the featurized data from script 1 (script1_out_table2_featurized.py)
dfTable2_wFeatures_wElementFrac = pd.read_csv('data/script1_out_table2_featurized.csv', low_memory=False)
# replace 'None' string values with np.nan
dfTable2_wFeatures_wElementFrac.replace('None', np.nan, inplace=True)
# make a list of all the columns that need to be cast to floats
lstFloatCols_table2 = dfTable2_wFeatures_wElementFrac.columns.tolist()
lstFloatCols_table2.remove('strComposition')
lstFloatCols_table2.remove('composition')
lstFloatCols_table2.remove('ElementFraction Exceptions')
lstFloatCols_table2.remove('ElementProperty Exceptions')
# cast all the columns to floats
dfTable2_wFeatures_wElementFrac[lstFloatCols_table2] = dfTable2_wFeatures_wElementFrac[lstFloatCols_table2].astype(float)

# no further cleaning is needed for the table 2 dataset - save it
dfTable2_wFeatures_wElementFrac.to_csv('data/script2_out_table2_clean.csv')


#%%
# CLEAN THE FEATURIZED DATA TO REMOVE DUPLICATES-----------------------------------------------------------------------
# remove duplicate compositions by selecting the lowest energy compound for every duplicate for the OQMD dataset
dfOQMD_clean_modelSelection = dfOQMD_wFeatures_wElementFrac.loc[dfOQMD_wFeatures_wElementFrac.groupby('strComposition')['energy_pa'].idxmin()]
# save the cleaned data to a csv file
dfOQMD_clean_modelSelection.to_csv('data/script2_out_oqmd_all_clean_modelSelection.csv')

# remove duplicate compositions by selecting the lowest energy compound for every duplicate for the bandgap dataset
dfBandgap_clean_framework= dfBandgap_wFeatures_wElementFrac.loc[dfBandgap_wFeatures_wElementFrac.groupby('strComposition')['energy_pa'].idxmin()]

#%%
# MAKE THE DATAFRAME FOR THE BANDGAP PREDICTION------------------------------------------------------------------------
# select the halogens from OQMD
dfOQMD_clean_framework = dfOQMD_clean_modelSelection[dfOQMD_clean_modelSelection['strComposition'].str.contains('F|Cl|Br|I')]
# select the bandgap values between 0eV/atom and 1.5eV/atom
dfOQMD_clean_framework = dfOQMD_clean_framework[(dfOQMD_clean_framework['bandgap_pa'] >= 0) & (dfOQMD_clean_framework['bandgap_pa'] <= 1.5)]
# save the cleaned data to a csv file
dfOQMD_clean_framework.to_csv('data/script2_out_oqmd_all_clean_framework.csv')
