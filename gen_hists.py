import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import arff
import os
import sys
import pandas as pd
import pickle

sns.set(style='white')
sns.set_palette('colorblind',10)

#a note - data was processed slightly differently for valence and arousal tasks (different window sizes),
#so we have two sets of timeseries data (one for each) and one file for all of the labels, per person

test = pickle.load(open('pickles/dev_1_arousal.pkl'))
cols = test.columns.values

for col in cols:
    try:
        fig, ax = plt.subplots(figsize=(12,9))
        to_plot = []
        to_plot_y = []
        for n in range(1,10):
            cur = pickle.load(open('pickles/train_' + str(n) + '_arousal.pkl'))
            to_plot.extend(cur[col].values)
            print len(cur[col])
            to_plot_y.extend(np.ones(len(cur[col]))*n)

        print np.shape(to_plot)
        print np.shape(to_plot_y)
        vp = sns.violinplot(x=to_plot_y, y=to_plot, vert=False, points=40, widths=1,
                                    showmeans=True, showextrema=True, showmedians=False)


        plt.ylabel('training individual', fontsize=13)
        plt.xlabel('feature distribution', fontsize=13)
        plt.title(col + ' over training users', y=1.03, fontsize=25)
        plt.savefig('./feature_violinplots/arousal_' + col + '.png')
    except:
        plt.close(fig)



