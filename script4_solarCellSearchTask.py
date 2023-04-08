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
from matplotlib.ticker import FormatStrFormatter

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

print('Packages imported successfully!\n')

#%%
# IMPORT THE FEATUREIZED DATA------------------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Importing featurized data for this task...\n')

dfTrain = pd.read_csv('data/script2_out_oqmd_solarCellSearchTask.csv', low_memory=False)
# shuffle the data
dfTrain = dfTrain.sample(frac=1, random_state=0)
print('Featurized training set imported')

# find the index of the first column that contains 'MagpieData'
intFirstMagpieCol_train = dfTrain.columns.get_loc(dfTrain.filter(regex='MagpieData').columns[0])
# find the index of the last column that contains 'MagpieData'
intLastMagpieCol_train = dfTrain.columns.get_loc(dfTrain.filter(regex='MagpieData').columns[-1])
# create a list of all the columns that contain 'MagpieData'
lstFeatureCols_train = dfTrain.columns[intFirstMagpieCol_train:intLastMagpieCol_train+1].tolist()
# make the feature set
dfTrain_x = dfTrain[lstFeatureCols_train]
print('Training feature set created')

# make the target set
srTrain_y = dfTrain['bandgap']
print('Test target set created')

# import the test set
dfTest = pd.read_csv('data/script2_out_table2_clean.csv')
print('\nFeaturized test set imported')

# find the index of the first column that contains 'MagpieData'
intFirstMagpieCol_test = dfTest.columns.get_loc(dfTest.filter(regex='MagpieData').columns[0])
# find the index of the last column that contains 'MagpieData'
intLastMagpieCol_test = dfTest.columns.get_loc(dfTest.filter(regex='MagpieData').columns[-1])
# create a list of all the columns that contain 'MagpieData'
lstFeatureCols_test = dfTest.columns[intFirstMagpieCol_test:intLastMagpieCol_test+1].tolist()
# make the feature set
dfTest_x = dfTest[lstFeatureCols_test]
print('Test feature set created')


srTest_y = dfTest['bandgap']
print('Test target set created')

lstTestCompositions = dfTest['composition'].tolist()

print('--------------------------------------------------------------------------------------------\n')

#%%
# INITIALIZE RANDOM FOREST PARAMETERS FROM THE WEKA DEFAULTS-----------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Initializing random forest parameters from the WEKA defaults...\n')

rfReg_n_estimators_weka = 100
rfReg_max_depth_weka = None
rfReg_min_samples_leaf_weka = 1
rfReg_min_samples_split_weka = 2
rfReg_max_features_weka = 'log2'

print('Random forest parameters from the WEKA defaults initialized')
print('--------------------------------------------------------------------------------------------\n')

# PREDICT THE BANDGAP OF THE TEST SET----------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Starting \'production run\' with 10 runs with different random seeds on the random forest model with WEKA hyperparameters...\n')

# fix the random seed so the same 10 random numbers are generated each time
np.random.seed(0)
# generate 10 random numbers between 0 and 1000
lstRandomSeeds = np.random.randint(0, 1000, 10).tolist()

# initialize a dataframe to store the predictions for each random seed
dfTest_yPred_weka = pd.DataFrame()
dfTest_yPred_weka['strComposition'] = lstTestCompositions

# initialize a counter to keep track of the number of runs
runCount = 1


for randomSeed in lstRandomSeeds:
      print('Run ', runCount, ' of 10')
      # make the random forest regressor
      rfReg_weka_temp = RandomForestRegressor(n_estimators = rfReg_n_estimators_weka,
                                              max_depth = rfReg_max_depth_weka,
                                              min_samples_leaf = rfReg_min_samples_leaf_weka,
                                              min_samples_split = rfReg_min_samples_split_weka,
                                              max_features = rfReg_max_features_weka,
                                              random_state = randomSeed)
      # fit the model to the training data
      rfReg_weka_temp_fit = rfReg_weka_temp.fit(dfTrain_x, srTrain_y)
      # predict the bandgap of the test set
      srTest_yPred_weka_temp = pd.Series(rfReg_weka_temp_fit.predict(dfTest_x), index=dfTest_x.index)

      # add the predictions as a new column to the dataframe
      dfTest_yPred_weka['Seed:' + str(randomSeed)] = srTest_yPred_weka_temp
      # increment the counter
      runCount += 1

# calculate the mean of the predictions for each random seed and make it a new column in the dataframe
dfTest_yPred_weka['mean'] = dfTest_yPred_weka.iloc[:, 1:].mean(axis=1)
# find the max of each prediction and make it a new column in the dataframe
dfTest_yPred_weka['max'] = dfTest_yPred_weka.iloc[:, 1:].max(axis=1)
# find the min of each prediction and make it a new column in the dataframe
dfTest_yPred_weka['min'] = dfTest_yPred_weka.iloc[:, 1:].min(axis=1)
# find the difference between the max and mean and make it a new column in the dataframe
dfTest_yPred_weka['max-mean'] = dfTest_yPred_weka['max'] - dfTest_yPred_weka['mean']
# find the difference between the mean and min and make it a new column in the dataframe
dfTest_yPred_weka['mean-min'] = dfTest_yPred_weka['mean'] - dfTest_yPred_weka['min']
#%%
# EVALUATION METRICS---------------------------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Calculating the mean absolute error...\n')

# calculate the mean absolute error for the weka model
wekaMAE = mean_absolute_error(srTest_y, dfTest_yPred_weka['mean'])
print('The mean absolute error of the model is: ', wekaMAE)

#%% 
# PLOT THE RESULTS-----------------------------------------------------------------------------------------------------

# setup subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor='w', edgecolor='k')

# plot the true bandgap values vs the strComposition
ax.scatter(range(len(srTest_y)), srTest_y, color='tab:blue', label='True bandgap')
# plot the predicted bandgap values vs the strComposition with error bars
ax.errorbar(range(len(srTest_y)), dfTest_yPred_weka['mean'], yerr=[dfTest_yPred_weka['mean-min'], dfTest_yPred_weka['max-mean']], color='tab:orange', label='Predicted bandgap', fmt='o', capsize=5)

# set the x and y axis labels
ax.set_xlabel('Composition', fontsize=12)
# make the x ticks the composition
ax.set_xticks(range(len(srTest_y)))
ax.set_xticklabels(lstTestCompositions, rotation=90, fontsize=12)
ax.set_ylabel('Bandgap (eV)', fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# set the title
ax.set_title('True vs Predicted Bandgap', fontsize=14)

# add a legend
ax.legend()
# set the layout
plt.tight_layout()
# save the plot
#plt.savefig('data/script4_out_plot.png', dpi=300)


#%%
# HYPERPARAMETER TUNING VIA GRIDSEARCHCV-------------------------------------------------------------------------------
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
hpt_rfReg = GridSearchCV(pipeline_rfReg,
                         rfRegParamsGrid,
                         scoring = 'neg_mean_absolute_error',
                         cv = 10,
                         return_train_score = True, 
                         n_jobs = -1)

# fit the grid search object to the training data
hpt_rfReg.fit(dfTrain_x, srTrain_y)

# make the results into a dataframe
dfHPT_rfReg = pd.DataFrame(hpt_rfReg.cv_results_)

# save the results of the grid search to a csv file
dfHPT_rfReg.to_csv('data/script4_out_hpt.csv', index=False)

print('Hyperparameter tuning complete')
print('--------------------------------------------------------------------------------------------\n')

#%%
# PLOT THE RESULTS OF THE GRID SEARCH----------------------------------------------------------------------------------
print('new cell:\n--------------------------------------------------------------------------------------------')
print('Printing the results of the grid search...\n')

# make a dataframe of the results
dfHPT_rfReg = pd.read_csv('data/script4_out_hpt.csv')

# extract the best hyperparameters
# take the first (lowest) of the n_estimators values that result in the best score
rfReg_n_estimators_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'param_randomforestregressor__n_estimators'])[0]
# take the first (lowest) of the max_depth values that result in the best score
rfReg_max_depth_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'param_randomforestregressor__max_depth'])[0]
# take the first (lowest) of the min_samples_leaf values that result in the best score
rfReg_min_samples_leaf_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'param_randomforestregressor__min_samples_leaf'])[0]
# take the first (lowest) of the min_samples_split values that result in the best score
rfReg_min_samples_split_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'param_randomforestregressor__min_samples_split'])[0]
# take the first (lowest) of the max_features values that result in the best score
rfReg_max_features_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'param_randomforestregressor__max_features'])[0]

print ('Random Forest Regressor Hyperparameter Tuning Complete\nThe hyperparameters that result in the best test score are:')
print('n_estimators: ', rfReg_n_estimators_best, 
      '\nmax_depth: ', rfReg_max_depth_best,
      '\nmin_samples_leaf: ', rfReg_min_samples_leaf_best,
      '\nmin_samples_split: ', rfReg_min_samples_split_best,
      '\nmax_features: ', rfReg_max_features_best)

# print the best score
rfReg_score_best = np.array(dfHPT_rfReg.loc[dfHPT_rfReg['rank_test_score'] == 1, 'mean_test_score'])[0]
print('Best mean absolute error: ', rfReg_score_best)

# # train a model with the best hyperparameters
# rfReg_best = RandomForestRegressor(n_estimators = rfReg_n_estimators_best,
#                                    max_depth = rfReg_max_depth_best,
#                                    min_samples_leaf = rfReg_min_samples_leaf_best,
#                                    min_samples_split = rfReg_min_samples_split_best,
#                                    max_features = rfReg_max_features_best,
#                                    random_state = 0)

print('--------------------------------------------------------------------------------------------\n')