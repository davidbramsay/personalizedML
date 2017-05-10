import arff
import os
import sys
import pandas as pd

#a note - data was processed slightly differently for valence and arousal tasks (different window sizes),
#so we have two sets of timeseries data (one for each) and one file for all of the labels, per person


recola_mapping = {
        'train_1':'P39',
        'train_2':'P23',
        'train_3':'P41',
        'train_4':'P46',
        'train_5':'P37',
        'train_6':'P16',
        'train_7':'P21',
        'train_8':'P25',
        'train_9':'P56',
        'test_1':'P38',
        'test_2':'P32',
        'test_3':'P53',
        'test_4':'P63',
        'test_5':'P59',
        'test_6':'P13',
        'test_7':'P47',
        'test_8':'P49',
        'test_9':'P20',
        'dev_1':'P45',
        'dev_2':'P26',
        'dev_3':'P64',
        'dev_4':'P34',
        'dev_5':'P42',
        'dev_6':'P65',
        'dev_7':'P30',
        'dev_8':'P19',
        'dev_9':'P28'
        }



def handle_recola_csv(file_path):
    if 'social' in file_path:
        df = pd.DataFrame.from_csv(file_path, index_col=False, sep=';')
        df = df.add_prefix('recola_' + file_path.split('/')[-2] + '_')
    elif 'emotion' in file_path:
        print '>>  IGNORE THIS FILE, THROW INTENTIONAL ERROR <<'
        df = 'x' #pd.DataFrame.from_csv(file_path, index_col=0, sep=';')
        df = 'x' #df.add_prefix('recola_' + file_path.split('/')[-2] + '_')
    else:
        df = pd.DataFrame.from_csv(file_path, index_col=0, sep=';')
        df = df.add_prefix('recola_' + file_path.split('/')[-2] + '_')


    print '...adding ' + file_path
    #print df.head()
    print 'found ' + str(len(list(df))) + ' length = ' + str(len(df))
    return df, (True if len(df) > 1 else False)



def handle_arff(file_path):
    data = arff.load(open(file_path), 'rb')
    df = pd.DataFrame.from_records(data['data'], columns=[d[0] for d in data['attributes']])
    df = df.rename(columns={'frametime':'time','frameTime':'time', 'GoldStandard':'GoldStandard_' + file_path.split('/')[-2]})
    df = df.set_index('time')

    print '...adding ' + file_path
    #print df.head()
    print 'found ' + str(len(list(df))) + ' length = ' + str(len(df))
    return df, (True if len(df) > 1 else False)




def combine_save_data():

    try:
        os.mkdir('./pickles')
        print '...made pickle directory'
    except:
        print '...found pickle directory'

    for (name, recola_name) in recola_mapping.items():
        print '----------------------------------------------------------------'
        print 'Combining files for ' + name

        return_df_arousal_timeseries = pd.DataFrame([])
        return_df_valence_timeseries = pd.DataFrame([])
        return_df_classes = pd.DataFrame([])

        for root, dirs, files in os.walk('./data'):
            for file in files:

                if (name in file or recola_name in file) and 'arff' in file:

                    path = os.path.join(root, file)
                    try:
                        new_data, is_timeseries = handle_arff(path)

                        if is_timeseries and 'valence' in path:
                            return_df_valence_timeseries = pd.concat([return_df_valence_timeseries, new_data], axis=1)
                        elif is_timeseries and 'arousal' in path:
                            return_df_arousal_timeseries = pd.concat([return_df_arousal_timeseries, new_data], axis=1)
                        elif is_timeseries:
                            return_df_valence_timeseries = pd.concat([return_df_valence_timeseries, new_data], axis=1)
                            return_df_arousal_timeseries = pd.concat([return_df_arousal_timeseries, new_data], axis=1)
                        else:
                            return_df_classes = pd.concat([return_df_classes, new_data], axis=1)

                    except:
                        print 'ERROR READING ' + os.path.join(root, file)

                if (name in file or recola_name in file) and 'csv' in file:

                    path = os.path.join(root, file)
                    try:
                        new_data, is_timeseries = handle_recola_csv(path)

                        if is_timeseries and 'valence' in path:
                            return_df_valence_timeseries = pd.concat([return_df_valence_timeseries, new_data], axis=1)
                        elif is_timeseries and 'arousal' in path:
                            return_df_arousal_timeseries = pd.concat([return_df_arousal_timeseries, new_data], axis=1)
                        elif is_timeseries:
                            return_df_valence_timeseries = pd.concat([return_df_valence_timeseries, new_data], axis=1)
                            return_df_arousal_timeseries = pd.concat([return_df_arousal_timeseries, new_data], axis=1)
                        else:
                            return_df_classes = pd.concat([return_df_classes, new_data], axis=1)

                    except:
                        print 'ERROR READING ' + os.path.join(root, file)

        print 'saving concatenated ' + name + ' to pickles...'
        return_df_valence_timeseries.to_pickle('./pickles/' + name + '_valence.pkl')
        return_df_arousal_timeseries.to_pickle('./pickles/' + name + '_arousal.pkl')
        return_df_classes.to_pickle('./pickles/' + name + '_classes.pkl')
        print 'Done.'

    print '...finished appending and saving files'



def add_labels_to_data(example_to_use = 'train_1'):

    df_ex_valence = pd.read_pickle('./pickles/' + example_to_use + '_valence.pkl')
    df_ex_arousal = pd.read_pickle('./pickles/' + example_to_use + '_arousal.pkl')
    df_ex_classes = pd.read_pickle('./pickles/' + example_to_use + '_classes.pkl')

    print "---------------------------------------------------------------------------------------------------------"
    print "---------------------------------------------------------------------------------------------------------"
    print "Valence timeseries data below, enter the number of the indices you'd like to predict, separated by commas"
    for i, name in enumerate(df_ex_valence.columns.values):
        print '(' + str(i) + ') ' + name
    input = raw_input("Enter comma separated values (INDEX of features you want to have as labels): ")
    v_inds = [int(a.strip()) for a in input.split(',')]

    v_names = {}
    for i in v_inds:
        v_names[df_ex_valence.columns.values[i]] = df_ex_valence.columns.values[i] + '_label'

    print "---------------------------------------------------------------------------------------------------------"
    print "---------------------------------------------------------------------------------------------------------"
    print "Arousal timeseries data below, enter the number of the indices you'd like to predict, separated by commas"
    for i, name in enumerate(df_ex_arousal.columns.values):
        print '(' + str(i) + ') ' + name
    input = raw_input("Enter comma separated values: ")
    a_inds = [int(a.strip()) for a in input.split(',')]

    a_names = {}
    for i in v_inds:
        a_names[df_ex_arousal.columns.values[i]] = df_ex_arousal.columns.values[i] + '_label'

    print "---------------------------------------------------------------------------------------------------------"
    print "---------------------------------------------------------------------------------------------------------"
    print "Class data below, enter the number of the indices you'd like to predict, separated by commas"
    for i, name in enumerate(df_ex_classes.columns.values):
        print '(' + str(i) + ') ' + name
    input = raw_input("Enter comma separated values: ")
    c_inds = [int(a.strip()) for a in input.split(',')]

    c_names = {}
    for i in c_inds:
        c_names[df_ex_classes.columns.values[i]] = df_ex_classes.columns.values[i] + '_label'

    print '--Name Array--'
    print v_names
    print a_names
    print c_names


    print "Updating all files to have 'label' in their name if we've noted it should be predicted..."
    for root, dirs, files in os.walk('./pickles'):
        for file in files:
            path = os.path.join(root, file)

            if 'valence' in file:
                df = pd.read_pickle(path)
                for i in v_inds:
                    df = df.rename(columns=v_names)
                df.to_pickle(path)
            elif 'arousal' in file:
                df = pd.read_pickle(path)
                for i in a_inds:
                    df = df.rename(columns=a_names)
                df.to_pickle(path)
            elif 'classes' in file:
                df = pd.read_pickle(path)
                for i in c_inds:
                    df = df.rename(columns=c_names)
                df.to_pickle(path)

    print "Done."



def add_classes_and_combine_pickles():

    #get all names
    names = set()
    for _, _, files in os.walk('./pickles'):
        for file in files:
            names.update(["_".join(file.split('_')[0:2])])


    for name in names:

        print 'restructuring ' + name
        #open classes, valence, and arousal pickles
        df_valence = pd.read_pickle('./pickles/' + name + '_valence.pkl')
        df_arousal = pd.read_pickle('./pickles/' + name + '_arousal.pkl')
        df_classes = pd.read_pickle('./pickles/' + name + '_classes.pkl')

        #add classes (through time) to valence and save
        df_final_valence = pd.concat([df_valence, df_classes.reindex(df_valence.index.values, method='pad')], axis=1)
        df_final_valence = df_final_valence.T.groupby(level=0).first().T # de-duplicate columns
        df_final_valence.to_pickle('./pickles/' + name + '_valence.pkl')

        #add classes (extend through time) to arousal and save
        df_final_arousal = pd.concat([df_arousal, df_classes.reindex(df_arousal.index.values, method='pad')], axis=1)
        df_final_arousal = df_final_arousal.T.groupby(level=0).first().T # de-duplicate columns
        df_final_arousal.to_pickle('./pickles/' + name + '_arousal.pkl')

        #re-label features that do not have recola, valence, or arousal in the name with _v _a for valence and arousal
        label_array = {}
        for i, col_name in enumerate(df_valence.columns.values):
            if not any([word in col_name for word in ['recola','valence','arousal']]):
                label_array[col_name] = col_name + '_v'

        df_valence = df_valence.rename(columns=label_array)

        label_array = {}
        for i, col_name in enumerate(df_arousal.columns.values):
            if not any([word in col_name for word in ['recola','valence','arousal']]):
                label_array[col_name] = col_name + '_a'

        df_arousal = df_arousal.rename(columns=label_array)

        #combine into one major pickle and save
        df_final_both = pd.concat([df_valence, df_arousal, df_classes.reindex(df_arousal.index.values, method='pad')], axis=1)
        df_final_both = df_final_both.T.groupby(level=0).first().T # de-duplicate columns
        df_final_both.to_pickle('./pickles/' + name + '_both.pkl')

        print 'finished.'


    print "Done."



def print_top(example_to_use = 'train_1'):

    df_ex_valence = pd.read_pickle('./pickles/' + example_to_use + '_valence.pkl')
    df_ex_arousal = pd.read_pickle('./pickles/' + example_to_use + '_arousal.pkl')
    df_ex_classes = pd.read_pickle('./pickles/' + example_to_use + '_classes.pkl')
    df_ex_classes = pd.read_pickle('./pickles/' + example_to_use + '_both.pkl')
    print 'Valence final:'
    for i, name in enumerate(df_ex_valence.columns.values):
        print '(' + str(i) + ') ' + name
        print df_ex_valence.head(n=20)
    print 'Arousal final:'
    for i, name in enumerate(df_ex_arousal.columns.values):
        print '(' + str(i) + ') ' + name
        print df_ex_arousal.head(n=20)
    print 'Classes final:'
    for i, name in enumerate(df_ex_classes.columns.values):
        print '(' + str(i) + ') ' + name
        print df_ex_classes.head(n=20)
    print 'Both final:'
    for i, name in enumerate(df_ex_both.columns.values):
        print '(' + str(i) + ') ' + name
        print df_ex_both.head(n=20)



def print_shape_and_vals(example_to_use = 'train_1'):

    df_ex_valence = pd.read_pickle('./pickles/' + example_to_use + '_valence.pkl')
    df_ex_arousal = pd.read_pickle('./pickles/' + example_to_use + '_arousal.pkl')
    df_ex_classes = pd.read_pickle('./pickles/' + example_to_use + '_classes.pkl')
    df_ex_both = pd.read_pickle('./pickles/' + example_to_use + '_both.pkl')

    print 'Valence final. shape ' + str(df_ex_valence.shape)
    for i, name in enumerate(df_ex_valence.columns.values):
        print '\t(' + str(i) + ') ' + name
    print 'Arousal final. shape ' + str(df_ex_arousal.shape)
    for i, name in enumerate(df_ex_arousal.columns.values):
        print '\t(' + str(i) + ') ' + name
    print 'Classes final. shape ' + str(df_ex_classes.shape)
    for i, name in enumerate(df_ex_classes.columns.values):
        print '\t(' + str(i) + ') ' + name
    print 'Both final. shape ' + str(df_ex_both.shape)
    for i, name in enumerate(df_ex_both.columns.values):
        print '\t(' + str(i) + ') ' + name



if __name__ == '__main__':
    combine_save_data()
    add_classes_and_combine_pickles()
    print_shape_and_vals()
