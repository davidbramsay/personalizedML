import arff
data = arff.load(open('data/features_ECG/arousal/dev_1.arff'), 'rb')

print data['attributes']
print data['relation']
print data['description']
#print data['data']

#data.data is a 2-D array, data[0], data[1] are timesteps; data[0:len(data)][0] is the timeseries data that corresponds to data.attributes[0]
