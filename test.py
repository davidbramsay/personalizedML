'''  TESTING ARFF LOAD OF INDIVIDUAL FEATURE FILES

import arff
data = arff.load(open('data/features_ECG/arousal/dev_1.arff'), 'rb')

print data['attributes']
print data['relation']
print data['description']
#print data['data']

#data.data is a 2-D array, data[0], data[1] are timesteps; data[0:len(data)][0] is the timeseries data that corresponds to data.attributes[0]

'''

###Test datafunc load


FM1_valence = ['recola_valence_FM1']
FM1_arousal = ['recola_arousal_FM1']
FM1_classes = ['recola_agreement_FM1','recola_dominance_FM1','recola_engagement_FM1','recola_performance_FM1','recola_rapport_FM1']


FM2_valence = ['recola_valence_FM2']
FM2_arousal = ['recola_arousal_FM2']
FM2_classes = ['recola_agreement_FM2','recola_dominance_FM2','recola_engagement_FM2','recola_performance_FM2','recola_rapport_FM2']

FM3_valence = ['recola_valence_FM3']
FM3_arousal = ['recola_arousal_FM3']
FM3_classes = ['recola_agreement_FM3','recola_dominance_FM3','recola_engagement_FM3','recola_performance_FM3','recola_rapport_FM3']

FF1_valence = ['recola_valence_FF1']
FF1_arousal = ['recola_arousal_FF1']
FF1_classes = ['recola_agreement_FF1','recola_dominance_FF1','recola_engagement_FF1','recola_performance_FF1','recola_rapport_FF1']

FF2_valence = ['recola_valence_FF2']
FF2_arousal = ['recola_arousal_FF2']
FF2_classes = ['recola_agreement_FF2','recola_dominance_FF2','recola_engagement_FF2','recola_performance_FF2','recola_rapport_FF2']

FF3_valence = ['recola_valence_FF3']
FF3_arousal = ['recola_arousal_FF3']
FF3_classes = ['recola_agreement_FF3','recola_dominance_FF3','recola_engagement_FF3','recola_performance_FF3','recola_rapport_FF3']

gold_valence = ['GoldStandard_valence']
gold_arousal = ['GoldStandard_arousal']










import data_funcs_avec as dataf
import combine_arffs as ca

#ca.print_shape_and_vals('train_1')
#ca.print_shape_and_vals('test_1')

print dataf.get_train_test_file_set()
a=dataf.DataLoader()
x,y= a.get_train_batch(5,20)
print x
print y
print x.shape
print y.shape
