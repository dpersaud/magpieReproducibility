#%%
# IMPORT PACKAGES------------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Monday, March 13th, 2023
Author: Daniel Persaud
Description: This script is used to generate the elemental and magpie properties for the datasets used in
the magpie reproducibility challenge and saves the dataframes to csv files for use in the next script.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction

print('Packages imported successfully\n')

#%%
# IMPORT THE RAW DATA AND CLEAN IT SO ONLY THE NECCESSARY COLUMNS ARE LEFT---------------------------------------------

# use pandas to read oqmd_all.data
dfOQMD = pd.read_csv('data/script1_in_oqmd_all.data', sep='  ', engine='python')
# change the comp column name to strComposition
dfOQMD = dfOQMD.rename(columns={'comp': 'strComposition'})
# drop entiries with 'None' as the entry for bandgap
#dfOQMD_bandgap = dfOQMD_bandgap.drop(dfOQMD_bandgap[dfOQMD_bandgap['bandgap'] == 'None'].index)
# cast all entries in the 'bandgap' column to floats
#dfOQMD_bandgap['bandgap'] = dfOQMD_bandgap['bandgap'].astype(float)
#dfOQMD_bandgap = dfOQMD_bandgap.loc[dfOQMD_bandgap.groupby('comp')['bandgap'].idxmin()]
print('OQMD data imported successfully')

# use pandas to read bandgap.data
dfBandgap = pd.read_csv('data/script1_in_bandgap.data', sep=' ', engine='python')
# change the comp column name to strComposition
dfBandgap = dfBandgap.rename(columns={'composition': 'strComposition'})
# drop entiries with 'None' as the entry for bandgap
#dfOQMD_bandgap = dfOQMD_bandgap.drop(dfOQMD_bandgap[dfOQMD_bandgap['bandgap'] == 'None'].index)
# cast all entries in the 'bandgap' column to floats
#dfOQMD_bandgap['bandgap'] = dfOQMD_bandgap['bandgap'].astype(float)
#dfOQMD_bandgap = dfOQMD_bandgap.loc[dfOQMD_bandgap.groupby('comp')['bandgap'].idxmin()]
print('Bandgap data imported successfully\n')

#%%
# GENERATE THE COMPOSITION OBJECT FROM THE STRING----------------------------------------------------------------------
# use the StrToComposition featurizer to convert the string composition to a composition object
dfOQMD = StrToComposition().featurize_dataframe(dfOQMD, "strComposition")
print('Composition object generated successfully for entire OQMD dataset')

dfBandgap = StrToComposition().featurize_dataframe(dfBandgap, "strComposition")
print('Composition object generated successfully for entire Bandgap dataset\n')

#%%
# GENERATE THE ELEMENTAL FRACTIONS-------------------------------------------------------------------------------------
# use the ElementFraction featurizer to generate the elemental fractions
dfOQMD = ElementFraction().featurize_dataframe(dfOQMD, col_id="composition", ignore_errors=True, return_errors=True)
print('Elemental fractions generated successfully for entire OQMD dataset')

dfBandgap = ElementFraction().featurize_dataframe(dfBandgap, col_id="composition", ignore_errors=True, return_errors=True)
print('Elemental fractions generated successfully for entire Bandgap dataset\n')
#%%
# GENERATE THE MAGPIE FEATURES-----------------------------------------------------------------------------------------
# use element property to featurize the data using magpie data
dfOQMD = ElementProperty.from_preset("magpie").featurize_dataframe(dfOQMD, col_id="composition", ignore_errors=True, return_errors=True)
print('Magpie features generated successfully for entire OQMD dataset')

dfBandgap = ElementProperty.from_preset("magpie").featurize_dataframe(dfBandgap, col_id="composition", ignore_errors=True, return_errors=True)
print('Magpie features generated successfully for entire Bandgap dataset\n')

#%%
# SAVE THE DATAFRAME TO A CSV FILE ------------------------------------------------------------------------------------
# save the dataframe to a csv file
dfOQMD.to_csv('data/script1_out_oqmd_all_featurized.csv', index=False)
dfBandgap.to_csv('data/script1_out_bandgap_featurized.csv', index=False)
print('Data saved successfully')