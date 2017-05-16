from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import metrics
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set(style='white')
sns.set_palette('colorblind',10)

maxlen= 80
max_features = 2000
num_samples_time = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
dropout = 0.2
recurrent_dropout = 0.2
epochs = 5
loss = 'mean_squared_error' #mean_absolute_error
optimizer = optimizers.SGD #'adam', #adagrad
learning_rate = 0.01
metrics = [metrics.mae, metrics.mse]
model_name = 'kerasTest1'
early_stopping_patience = 2
LSTM_layers = [128, 128]
num_labels = 2 #num things to label/predict
min_max_scale_labels = True

print('Loading data...')

X_train =
y_train =
X_val = X_test
y_val = y_test

if min_max_scale_labels:
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(np.concatenate((y_train, y_val, y_test), axis=0))
    y_train = scaler.transform(y_train)
    y_val = scaler.transform(y_val)
    y_test = scaler.transform(y_test)

print('Build model...')
model = Sequential()

#one LSTM layer desired
if len(LSTM_layers) == 1:
    model.add(LSTM(LSTM_layers[0],
                input_shape=(timesteps, data_dim),
                dropout=dropout, recurrent_dropout=recurrent_dropout))

#multiple LSTM layers desired
else:
    model.add(LSTM(LSTM_layers[0], return_sequences=True,
                input_shape=(timesteps, data_dim),
                dropout=dropout, recurrent_dropout=recurrent_dropout))

    for i in range(1, len(LSTM_layers)-1):
        model.add(LSTM(LSTM_layers[i], return_sequences=True,
                  dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(LSTM(LSTM_layers[-1], dropout=dropout, recurrent_dropout=recurrent_dropout))

#add a dense layer at the end, sigmoid if we've scaled our labels between 0 and 1
if min_max_scale_labels:
    model.add(Dense(num_labels), activation='sigmoid')
else:
    model.add(Dense(num_labels))


# try using different optimizers and different optimizer configs
model.compile(loss=loss,
              optimizer=optimizer(lr=learning_rate),
              metrics=metrics)


print('Training...')

#early stopping if we get no better val_loss for 3 epochs in a row
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_val, y_val),
          callbacks=[early_stopping])

print('Testing...')

#evaluate using our metrics
scores = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

print('-'*3 + model.metric_names + '-'*3)
print(scores)

#predict on the test set
y_pred = model.predict(X_test, batch_size=1)

if min_max_scale_labels:
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)


print('Saving...')

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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(model_name +' Model Loss (w/early stopping)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Actual', 'Prediction'], loc='upper left')
plt.savefig('./keras/' + model_name + '_prediction.png')

#save model
model.save('./keras/' + model_name + '.h5')

#save data
save_dict = {}
save_dict['params'] = { 'maxlen': maxlen,
                        'batch_size' : batch_size,
                        'dropout' : dropout,
                        'recurrent_dropout' : recurrent_dropout,
                        'epochs' : epochs,
                        'loss' : loss,
                        'optimizer' : optimizer,
                        'learning_rate' : learning_rate,
                        'metrics' : metrics,
                        'model_name' : model_name,
                        'early_stopping_patience' : patience
                        }

save_dict['scores'] = scores
save_dict['score_names'] = score_names

save_dict['history'] = history

save_dict['y_pred'] = y_pred
save_dict['y_target'] = y_test


pickle.dump(save_dict, open( model_name + "_data.p", "wb" ) )

