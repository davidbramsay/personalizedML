""" This file provides functions for loading data from a saved
    .csv file.

    The code assumes that the file contains at least one column
    with 'label' in the name, containing an outcome we are trying
    to classify. E.g. if we are trying to predict if someone is
    happy or not, we might have a column 'happy_label' that has
    values that are either 0 or 1.

    We also assume the file contains a column named 'dataset'
    containing the dataset the example belongs to, which can be
    either 'Train', 'Val', or 'Test'. Remember, it is important
    to never train or choose parameters based on the test set!
"""

import pandas as pd
import numpy as np

#TODO
# -store normalization based on full training set
# -open, normalize, and batch all training/test/validation data
# -

classes = ['recola_agreement_FM1','recola_dominance_FM1','recola_engagement_FM1','recola_performance_FM1','recola_rapport_FM1',
           'recola_agreement_FM2','recola_dominance_FM2','recola_engagement_FM2','recola_performance_FM2','recola_rapport_FM2',
           'recola_agreement_FM3','recola_dominance_FM3','recola_engagement_FM3','recola_performance_FM3','recola_rapport_FM3',
           'recola_agreement_FF1','recola_dominance_FF1','recola_engagement_FF1','recola_performance_FF1','recola_rapport_FF1',
           'recola_agreement_FF2','recola_dominance_FF2','recola_engagement_FF2','recola_performance_FF2','recola_rapport_FF2',
           'recola_agreement_FF3','recola_dominance_FF3','recola_engagement_FF3','recola_performance_FF3','recola_rapport_FF3']


class DataLoader:
    def __init__(self, peopleSplit='ignoreTest', dataset='both', ignoreFeats = [],
                includeClasses=True, labels=['GoldStandard_valence','GoldStandard_arousal']):

        self.load_and_process_data(peopleSplit=peopleSplit, dataset=dataset, ignoreFeats=ignoreFeats,
                                   includeClasses=includeClasses, labels=labels)

    def load_and_process_data( self,
                            peopleSplit='ignoreTest', #could be asProvided, to use included testset data, which we don't have
                            dataset='both', #could be 'arousal' or 'valence'
                            includeClasses=True,
                            labels=['GoldStandard_valence','GoldStandard_arousal'], #things to predict
                            ignoreFeats = [], #columns (features/labels) to ignore
                            suppress_output=False ):

        train_files, val_files, test_files = get_train_test_file_set(peopleSplit, dataset)

        if not includeClasses:
            ignoreFeats.extend(classes)

        self.ignoreFeats=ignoreFeats

        train_dict, val_dict, test_dict = {}, {}, {}

        for i, f in enumerate(train_files):
            train_dict[i] = pd.read_pickle(f)
        for i, f in enumerate(val_files):
            val_dict[i] = pd.read_pickle(f)
        for i, f in enumerate(test_files):
            test_dict[i] = pd.read_pickle(f)

        self.train_df = pd.concat(train_dict.values(),axis=0,keys=train_dict.keys())
        self.val_df = pd.concat(val_dict.values(),axis=0,keys=val_dict.keys())
        self.test_df = pd.concat(test_dict.values(),axis=0,keys=test_dict.keys())

        #remove instance_name variables
        instance_cols = [x for x in self.train_df if 'instance' in x.lower()]

        for col in instance_cols:
            del self.train_df[col]
            del self.val_df[col]
            del self.test_df[col]

        #get wanted features and labels
        self.wanted_feats = [x for x in self.train_df.columns.values if x not in labels and x not in ignoreFeats]
        self.wanted_labels = [y for y in self.train_df.columns.values if y in labels and y not in ignoreFeats]

        print 'loaded training set, shape:' + str(self.train_df.shape)
        print 'loaded val set, shape:' + str(self.val_df.shape)
        print 'loaded test set, shape:' + str(self.test_df.shape)
        print ' - ignoring: ' + str(ignoreFeats)
        print ' - features : %s, ignored: %s' % (len(self.wanted_feats), (len(self.train_df.columns.values) - len(self.wanted_feats) - len(self.wanted_labels)))
        print ' - labels: ' + str(self.wanted_labels)

        #self.normalize_fill_df()

        self.num_outputs = len(self.wanted_labels)

        self.train_X, self.train_Y = get_matrices_for_dataset(self.train_df, self.wanted_feats,
                                                            self.wanted_labels, len(self.wanted_labels)==1)

        self.val_X, self.val_Y = get_matrices_for_dataset(self.val_df, self.wanted_feats,
                                                        self.wanted_labels, len(self.wanted_labels)==1)

        self.test_X, self.test_Y = get_matrices_for_dataset(self.test_df, self.wanted_feats,
                                                            self.wanted_labels, len(self.wanted_labels)==1)


    def get_train_batch(self, batch_size, num_time_steps):
        #need to grab batch_size num_time_step slices of data from different people
        leng, width, _ = self.train_X.shape
        print leng, width

        return_train_x = np.empty([batch_size, num_time_steps, len(self.wanted_feats)])
        return_train_y = np.empty([batch_size, num_time_steps, len(self.wanted_labels)])
        print return_train_x.shape
        print return_train_y.shape

        for i in range(batch_size):
            idx = np.random.choice(leng)
            idy = np.random.choice(width-num_time_steps+1)

            return_train_x[i]=self.train_X[idx, idy:idy+num_time_steps, :]
            return_train_y[i]=self.train_Y[idx, idy:idy+num_time_steps, :]

        return return_train_x, return_train_y

    def get_val_data(self):
        return self.val_X, self.val_Y

    def get_feature_size(self):
        return np.shape(self.train_X)[2]

    def normalize_fill_df(self):
        self.normalize_columns()

        self.train_df = self.train_df.fillna(0) #if dataset is already filled, won't do anything
        self.val_df = self.val_df.fillna(0) #if dataset is already filled, won't do anything
        self.test_df = self.test_df.fillna(0) #if dataset is already filled, won't do anything

    def normalize_columns(self):

        #only use training data, apply to all data
        for feat in self.wanted_feats:

            train_mean = np.mean(self.train_df[feat].dropna().tolist())
            train_std = np.std(self.train_df[feat].dropna().tolist())
            zscore = lambda x: (x - train_mean) / train_std

            self.train_df[feat] = self.train_df[feat].apply(zscore)
            self.val_df[feat] = self.val_df[feat].apply(zscore)
            self.test_df[feat] = self.test_df[feat].apply(zscore)


def get_matrices_for_dataset(df, wanted_feats, wanted_labels, single_output=False):

    ind1 = len(set(df.index.get_level_values(0)))
    ind2 = len(set(df.index.get_level_values(1)))

    X = df[wanted_feats].astype(float).values
    X = np.reshape(X, (ind1, ind2, -1))

    if single_output:
        y = df[wanted_labels[0]].values
    else:
        y = df[wanted_labels].values

    y = np.reshape(y, (ind1, ind2, -1))

    X = convert_matrix_tf_format(X)
    y = np.atleast_2d(np.asarray(y))

    return X,y

def convert_matrix_tf_format(X):
    X = np.asarray(X)
    X = X.astype(np.float64)
    return X



def get_train_test_file_set(default='ignoreTest', dataset='both'):
    '''have to ignore test set because we don't have the valence/arousal values'''
    '''dataset can be 'valence', 'arousal', or 'both'  '''

    if default == 'ignoreTest':
        training = ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'train_7', 'train_8']
        validation = ['train_9', 'dev_1', 'dev_2', 'dev_3', 'dev_4']
        test = ['dev_9','dev_8','dev_7','dev_6','dev_5']

    else:

        training = ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'train_7', 'train_8', 'train_9']
        validation = ['dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_5', 'dev_6', 'dev_7', 'dev_8', 'dev_9']
        test = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8', 'test_9']

    for i in range(len(training)):
        training[i] =  './pickles/' + training[i] + '_' + dataset + '.pkl'

    for i in range(len(validation)):
        validation[i] = './pickles/' + validation[i] + '_' + dataset + '.pkl'

    for i in range(len(test)):
        test[i] = './pickles/' + test[i] + '_' + dataset + '.pkl'


    return set(training), set(validation), set(test)
