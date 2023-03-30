#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Monday, March 17th, 2023
Author: Daniel Persaud
Description:    This script will hyperparameter tune a random forest model using 10-fold random CV via GridSearchCV to 
                obtain the best model parameters and in so doing, the best possible 10-fold random cross validation 
                score to compare to the original paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print('Packages imported successfully!')

#%%
# IMPORT THE FEATUREIZED DATA------------------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Importing featurized data for this task...\n')
dfTrain = pd.read_csv('data/script2_out_oqmd_modelSelectionTask.csv', low_memory=False)
# shuffle the data
dfTrain = dfTrain.sample(frac=1, random_state=0)
print('Featurized data imported')

# find the index of the first column that contains 'MagpieData'
intFirstMagpieCol = dfTrain.columns.get_loc(dfTrain.filter(regex='MagpieData').columns[0])
# find the index of the last column that contains 'MagpieData'
intLastMagpieCol = dfTrain.columns.get_loc(dfTrain.filter(regex='MagpieData').columns[-1])
# create a list of all the columns that contain 'MagpieData'
lstFeatureCols = dfTrain.columns[intFirstMagpieCol:intLastMagpieCol+1].tolist()

# make a string of the target column name
strTarget = 'bandgap'

# make the feature set
dfTrain_x = dfTrain[lstFeatureCols]
print('Feature set created')
# make the target set
dfTrain_y = dfTrain[strTarget]
print('Target set created')
print('--------------------------------------------------------------------------------------------\n')

# %%
# HYPERPARAMETER TUNING VIA GRIDSEARCHCV > BEST MAE POSSIBLE IN 10-FOLD RANDOM CV--------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Hyperparameter tuning via GridSearchCV...\n')

# set the random state for reproducibility
np.random.seed(0)


# initialize the model (in a pipeline to scale the data such that data leakage does not occur)
pipeline_rfReg = make_pipeline(StandardScaler(),RandomForestRegressor(random_state=0))

# make lists for possible values for hyperparameters
rfReg_n_estimators_totest = [1, 5, 10, 50, 100, 500, 1000]  # number of trees in forest
rfReg_max_depth_totest = [1, 5, 10, 50, 100, 500, 1000]     # the maximum depth of each tree
rfReg_min_sample_leaf_totest = [1, 2, 5, 10]                # minimum number of samples required to be at a leaf node
rfReg_min_sample_split_totest = [2, 4, 8, 10]               # minimum number of samples required to be at a leaf node
rfReg_max_features_totest = [1/4, 1/3, 1/2, 2/3, 3/4, 1]    # number of features to consider when looking for the best split

# put those lists into a dictionary to pass into GridSearchCV
rfRegParamsGrid = {'randomforestregressor__n_estimators'        :rfReg_n_estimators_totest,
                   'randomforestregressor__max_depth'           :rfReg_max_depth_totest,
                   'randomforestregressor__min_samples_leaf'    :rfReg_min_sample_leaf_totest,
                   'randomforestregressor__min_samples_split'   :rfReg_min_sample_split_totest,
                   'randomforestregressor__max_features'        :rfReg_max_features_totest}

# make the grid search object - to comment out once the grid search has been run
hpt_rfReg = GridSearchCV(pipeline_rfReg, rfRegParamsGrid, scoring='neg_mean_absolute_error',
                         cv = 10, return_train_score = True, n_jobs = -1)

# fit the grid search object to the training data
hpt_rfReg.fit(dfTrain_x, dfTrain_y)

# make the results into a dataframe
dfhpt_rfReg = pd.DataFrame(hpt_rfReg.cv_results_)

# save the results of the grid search to a csv file
dfhpt_rfReg.to_csv('data/script3_out_rfCV.csv', index=False)

print('Hyperparameter tuning complete')
print('--------------------------------------------------------------------------------------------\n')

#%% 
# READ THE RESULTS OF THE GRID SEARCH AND PRINT THE BEST SCORE--------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Reading the results of the grid search and printing the best score...\n')

# read the results of the grid search
dfhpt_rfReg = pd.read_csv('data/script3_out_rfCV.csv', low_memory=False)

# find the index of the row with the best score
intBestScoreIndex = dfhpt_rfReg['rank_test_score'].idxmin()

# print the best score
print('The best mean absolute error is: ', dfhpt_rfReg['mean_test_score'][intBestScoreIndex])

