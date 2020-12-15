import opensmile, json

def featurize_opensmile(wav_file):

	# initialize features and labels
	labels=list()
	features=list()

	# extract LLD 
	smile_LLD = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
	)

	y_LLD = smile_LLD.process_file(wav_file)

	labels_LLD=list(y_LLD)

	for i in range(len(labels_LLD)):
		features.append(list(y_LLD[labels_LLD[i]]))
		labels.append(labels_LLD[i])

	smile_LLD_deltas = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,

	)

	y_LLD_deltas = smile_LLD_deltas.process_file(wav_file)

	labels_LLD_deltas=list(y_LLD_deltas)

	for i in range(len(labels_LLD_deltas)):
		features.append(list(y_LLD_deltas[labels_LLD_deltas[i]]))
		labels.append(labels_LLD_deltas[i])

	smile_functionals = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.Functionals,
	)

	y_functionals = smile_functionals.process_file(wav_file)

	labels_y_functionals=list(y_functionals)

	for i in range(len(labels_y_functionals)):
		features.append(list(y_functionals[labels_y_functionals[i]]))
		labels.append(labels_y_functionals[i])

	return features, labels

# features, labels = featurize_opensmile('test.wav')

# print(labels)
# data=dict()
# for i in range(len(labels)):
# 	data[labels[i]]=features[i]

# g=open('test.json','w')
# json.dump(data,g)
# g.close()

# print(features)
# print(labels)