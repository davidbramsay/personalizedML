from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import metrics
import numpy as np
import seaborn as sns
import data_funcs_avec as dataf

sns.set(style='white')
sns.set_palette('colorblind',10)

#----- convenience definitions for ignoreFeats and labels below

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

#-----

#many-to-many : predict full sequence in time
#many-to-many : predict small snippets, but use state so it remembers for each batch, and concatenate
#many-to-many : predict small snippets, use state to remember each batch, 1-D batch with reset in between each
#many-to-one : use sequence to predict last value, sliding window


#----- PARAMETERS TO SET -----#

peopleSplit='quickTesting'#'ignoreTest' # or 'asProvided' if we had labled test data
dataset='both' #'both', 'arousal', or 'valence'
ignoreFeats = [] #features to ignore completely in this test
includeClasses=True #include or disclude social ratings (dominance, rapport..)
labels=['GoldStandard_valence','GoldStandard_arousal'] #things to predict

batch_size = 32
timesteps = 200

num_batches_per_epoch = 10
num_epochs = 2

dropout = 0.2
recurrent_dropout = 0.2
loss = 'mean_absolute_error' #'mean_squared_error' #mean_absolute_error
optimizer = optimizers.rmsprop #optimizers.SGD #'adam', #adagrad
learning_rate = 0.001
metrics = [metrics.mae, metrics.mse]
model_name = 'kerasTest1'
early_stopping_patience = 2
LSTM_layers = [128]
ret_sequence = True #leave this to get a timeseries


#input is (batch_size, timesteps, num_features)  (first dim, batch_size, can be None=any size)
#output is (batch_size, timesteps, num_outputs)

#can do stateful and have batch_size = num training people

#need return_sequence = True for all middle layers, last layer if we want a sequence

print('Loading data...')
dataGen = dataf.DataLoader(peopleSplit, dataset, ignoreFeats, includeClasses, labels)

tempx, tempy = dataGen.get_train_batch_rand(batch_size, timesteps)
num_features = tempx.shape[2]
num_labels = tempy.shape[2]

print('num features, labels')
print(num_features)
print(num_labels)

X_val = dataGen.val_X
y_val = dataGen.val_Y
X_test = dataGen.test_X
y_test = dataGen.test_Y

print('shape of X_val, Y_val, X_test, and Y_test')
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

print('Build model...')
model = Sequential()

#one LSTM layer desired
if len(LSTM_layers) == 1:
    model.add(LSTM(LSTM_layers[0], return_sequences=ret_sequence,
                input_shape=(timesteps, num_features),
                dropout=dropout, recurrent_dropout=recurrent_dropout))

#multiple LSTM layers desired
else:
    model.add(LSTM(LSTM_layers[0], return_sequences=True,
                input_shape=(timesteps, num_features),
                dropout=dropout, recurrent_dropout=recurrent_dropout))

    for i in range(1, len(LSTM_layers)-1):
        model.add(LSTM(LSTM_layers[i], return_sequences=True,
                  dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(LSTM(LSTM_layers[-1], return_sequences=ret_sequence,
        dropout=dropout, recurrent_dropout=recurrent_dropout))

#add a dense layer at the end
if ret_sequence:
    model.add(TimeDistributed(Dense(num_labels)))
else:
    model.add(Dense(num_labels))

# try using different optimizers and different optimizer configs
model.compile(loss=loss,
              optimizer=optimizer(lr=learning_rate, clipnorm=1.),
              metrics=metrics)


print('Training...')

#early stopping if we get no better val_loss for 3 epochs in a row
early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

'''
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(X_val, y_val),
          callbacks=[early_stopping])
'''

print('reshape validation')
print( X_val.shape )
print( y_val.shape )
valX = np.reshape(np.array(X_val)[:,:X_val.shape[1]//timesteps*timesteps,:], (-1, timesteps, num_features))
valy = np.reshape(np.array(y_val)[:,:y_val.shape[1]//timesteps*timesteps,:], (-1, timesteps, num_labels))
print( valX.shape )
print( valy.shape )

history = model.fit_generator(dataGen.get_train_generator_seq(batch_size,timesteps),
                            steps_per_epoch = num_batches_per_epoch,
                            epochs = num_epochs,
                            validation_data=(valX, valy),
                            callbacks=[early_stopping])


print('Testing...')

#evaluate using our metrics

testX = np.reshape(np.array(X_test)[:,:X_test.shape[1]//timesteps*timesteps,:], (-1, timesteps, num_features))
testy = np.reshape(np.array(y_test)[:,:y_test.shape[1]//timesteps*timesteps,:], (-1, timesteps, num_labels))

scores = model.evaluate(testX, testy,
                            batch_size=X_test.shape[0])

print('-'*3 + str(model.metrics_names) + '-'*3)
print(scores)

#predict on the test set
y_pred = model.predict(testX, batch_size=X_test.shape[0])

print('Saving...')

#save model
model.save('./keras/' + model_name + '.h5')

#save data
save_dict = {}
save_dict['params'] = { 'batch_size' : batch_size,
                        'timesteps' : timesteps,
                        'num_batches_per_epoch' : num_batches_per_epoch,
                        'num_epochs' : num_epochs,
                        'num_features' : num_features,
                        'num_labels' : num_labels,
                        'peopleSplit' : peopleSplit,
                        'dataset' : dataset,
                        'ignoreFeats' : ignoreFeats,
                        'includeClasses' : includeClasses,
                        'labels' : labels,
                        'LSTM_layers' : LSTM_layers,
                        'ret_sequence' : ret_sequence,
                        'dropout' : dropout,
                        'recurrent_dropout' : recurrent_dropout,
                        'loss' : loss,
                        'optimizer' : optimizer,
                        'learning_rate' : learning_rate,
                        'metrics' : metrics,
                        'model_name' : model_name,
                        'early_stopping_patience' : early_stopping_patience
                        }

save_dict['scores'] = scores
#save_dict['score_names'] = score_names

save_dict['history'] = history

save_dict['y_pred'] = y_pred
save_dict['y_target'] = testy


pickle.dump(save_dict, open( model_name + "_data.p", "wb" ) )


#save visuals

#visual of network
plot_model(model, to_file = './keras/' + model_name + '_model.png')

#visual of training/val loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(model_name +' Model Loss (w/early stopping)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('./keras/' + model_name + '_loss.png')

#visual of predicted vs actual label values
plt.plot(y_test)
plt.plot(y_pred)
plt.title(model_name +' Prediction vs. Actual')
plt.ylabel('valence/arousal value')
plt.xlabel('time (steps)')
plt.legend(['Actual', 'Prediction'], loc='upper left')
plt.savefig('./keras/' + model_name + '_prediction.png')


'''

# Train a Stateful LSTM
# ---------------------
# get *all* data, flatten it out into one long array of batchsize and timestep length = 1
# (here num_features also equals 1)
# when the state shouldn't be propogated (maxlen is the length of each
# sequence), we manually call reset_states to stop propogating time
# dependencies.

#this should work for larger batch sizes as well if modulo(sequence length, batch size) == 0

max_len = 20
class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs={}):
        if self.counter % max_len == 0:
            self.model.reset_states()
        self.counter += 1

x = np.expand_dims(np.expand_dims(X_train.flatten(), axis=1), axis=1)
y = np.expand_dims(np.array([[v] * max_len for v in y_train.flatten()]).flatten(), axis=1)
model.fit(x, y, callbacks=[ResetStatesCallback()], batch_size=1, shuffle=False)
'''
