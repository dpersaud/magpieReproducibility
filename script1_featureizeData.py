#%%
# IMPORT PACKAGES------------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Tuesday, March 28th, 2023

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

print('Packages imported successfully!\n')

#%%
# IMPORT THE RAW DATA--------------------------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Importing raw data...\n')

# use pandas to read oqmd.data
dfOQMD = pd.read_csv('data/script1_in_oqmd.data', sep='  ', engine='python')
# change the comp column name to strComposition
dfOQMD = dfOQMD.rename(columns={'comp': 'strComposition'})
print('OQMD data imported')

# use pandas to read bandgap.data
dfBandgap = pd.read_csv('data/script1_in_bandgap.data', sep=' ', engine='python')
# change the comp column name to strComposition
dfBandgap = dfBandgap.rename(columns={'composition': 'strComposition'})
print('Bandgap data imported')

# make a dataframe for the 'test' dataset - table 2 from the original paper
dfTable2 = pd.DataFrame({'strComposition' : ['ScHg4Cl7',
                                             'V2Hg3Cl7',
                                             'Mn6CCl8',
                                             'Hf4S11Cl2',
                                             'VCu5Cl9'],
                         'bandgap' : [1.26,
                                      1.16,
                                      1.28,
                                      1.11,
                                      1.19]})
print('Test data created')

print('\nRaw data imported / created')
print('--------------------------------------------------------------------------------------------\n')
                                           

#%%
# GENERATE THE COMPOSITION OBJECT FROM THE STRING----------------------------------------------------------------------
# use the StrToComposition featurizer to convert the string composition to a pymatgen.core.composition.Composition object
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Generating the composition object from the string...\n')

dfOQMD = StrToComposition().featurize_dataframe(dfOQMD, "strComposition")
print('Composition object generated for entire OQMD dataset')

dfBandgap = StrToComposition().featurize_dataframe(dfBandgap, "strComposition")
print('Composition object generated for entire Bandgap dataset')

dfTable2 = StrToComposition().featurize_dataframe(dfTable2, "strComposition")
print('Composition object generated for table 2')

print('\nComposition object generated successfully for all datasets')
print('--------------------------------------------------------------------------------------------\n')

#%%
# GENERATE THE ELEMENTAL FRACTIONS-------------------------------------------------------------------------------------
# use the ElementFraction featurizer to generate the elemental fractions based on the pymatgen.core.composition.Composition object
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Generating the elemental fractions...\n')

dfOQMD = ElementFraction().featurize_dataframe(dfOQMD, col_id="composition", ignore_errors=True, return_errors=True)
print('Elemental fractions generated for entire OQMD dataset')

dfBandgap = ElementFraction().featurize_dataframe(dfBandgap, col_id="composition", ignore_errors=True, return_errors=True)
print('Elemental fractions generated for entire Bandgap dataset')

dfTable2 = ElementFraction().featurize_dataframe(dfTable2, col_id="composition", ignore_errors=True, return_errors=True)
print('Elemental fractions generated for table 2')

print('\nElemental fractions generated successfully for all datasets')
print('--------------------------------------------------------------------------------------------\n')
#%%
# GENERATE THE MAGPIE FEATURES-----------------------------------------------------------------------------------------
# use element property to generate the magpie features from the pymatgen.core.composition.Composition object
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Generating the magpie features...\n')

dfOQMD = ElementProperty.from_preset("magpie").featurize_dataframe(dfOQMD, col_id="composition", ignore_errors=True, return_errors=True)
print('Magpie features generated for entire OQMD dataset')

dfBandgap = ElementProperty.from_preset("magpie").featurize_dataframe(dfBandgap, col_id="composition", ignore_errors=True, return_errors=True)
print('Magpie features generated for entire Bandgap dataset')

dfTable2 = ElementProperty.from_preset("magpie").featurize_dataframe(dfTable2, col_id="composition", ignore_errors=True, return_errors=True)
print('Magpie features generated for table 2')

print('\nMagpie features generated for all datasets')
print('--------------------------------------------------------------------------------------------\n')

#%%
# SAVE THE DATAFRAME TO A CSV FILE ------------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Saving the dataframes to csv files...\n')

dfOQMD.to_csv('data/script1_out_oqmd_featurized.csv', index=False)
dfBandgap.to_csv('data/script1_out_bandgap_featurized.csv', index=False)
dfTable2.to_csv('data/script1_out_table2_featurized.csv', index=False)

print('Data saved successfully')
print('--------------------------------------------------------------------------------------------\n')