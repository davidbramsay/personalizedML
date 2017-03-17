## how to (so far)

the combine_arff.py script works as follows:

AVEC data must be in ./data/

The script will run through all arffs and csvs in ./data and combine them into
panda dataframes per user, which are saved in ./pickles.  HR/EDA etc values are
windowed differently when predicting arousal and valence.  We create one
dataframe for arousal timeseries data, one dataframe for valence timeseries
data, and one dataframe for non-timeseries labels (each are saved under the
individual's name, i.e. 'train_1').

The second part of the script will update all column values that you select to
have '_label' added to their column name.  This can be used with Natasha's NN
example to differentiate features from final values to predict.

This should make the data easy to work with and train per user. 

