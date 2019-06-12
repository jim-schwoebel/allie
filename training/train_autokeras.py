from autokeras import MlpModule
from autokeras.backend.torch.loss_function import classification_loss
from autokeras.backend.torch.loss_function import regression_loss
from autokeras.nn.metric import Accuracy
from sklearn.model_selection import train_test_split

# pre processing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from pandas import Series
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
import numpy as np
from keras.utils import to_categorical

def transform_data(features, labels, normalize):

	dimension=len(features[0])

	# integer encode
	values = np.array(labels)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	num_classes=np.amax(integer_encoded)+1
	new_labels=to_categorical(integer_encoded, num_classes)     
	labels=new_labels

	if normalize == True:
		# PREPROCESS THIS DATA TO PROPER FORMAT
		feature_length=len(features[0])
		new_features=list()
		all_features=np.array([])
		# get the min and max all feature matrices
		print('appending all features into 1 numpy array')
		for i in range(len(features)):
		   all_features=np.append(all_features,np.array(features[i]))
		##    # check for list features to debug...
		##    for i in range(len(all_features)):
		##        if type(all_features[i])==type(list()):
		##            print(i)

		# normalize [0,1]
		values = all_features.reshape((len(all_features), 1))
		scaler = MinMaxScaler().fit(values)

	print('looping through all features and making list')
	for i in range(len(features)):
	   series = Series(features[i])
	   values= series.values
	   values = values.reshape((len(values), 1))
	   normalized = scaler.transform(values)
	   # one hot encoding via sklearn library
	   feature=np.concatenate(normalized)
	   new_features.append(feature)

	# ensure dimension match up!
	# now iterate through all labels and features to remove samples not of same length
	new_features=list()
	new_labels=list()
	print('iterating through features and making sure they are the same length')
	for i in range(len(features)):
	    if len(features[i])==dimension:
	        new_features.append(np.array(features[i]))
	        new_labels.append(labels[i])
	    else:
	        pass

	features=np.array(new_features)
	labels=np.array(new_labels)

	##########################################################################################################
	##                                      WHAT INPUT DATA LOOKS LIKE:                                     ##
	##########################################################################################################
	##    features = [-309.13900050822417, 62.994113729391806, -413.79101175121536, -126.44824022654625,    ##
	##                145.68530325221258, 36.63236047297892, 46.81091824978685, 224.72785358499007,         ##
	##                -37.22843646027476, 29.019041509665218, -132.72131653263105, 17.649062326512716,      ##
	##                ...                                                                                   ##
	##                -1.4978552705426866, 3.407496547510848, -5.401921783350719, 4.815302136763165,        ##
	##                -0.1618493623403922, 4.339649083555491, -7.4368054086188335, 6.86717455737755]        ##
	##                                                                                                      ##
	##    labels = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']            ##
	##########################################################################################################


	##  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ## 

	##########################################
	##    DATA SHOULD NOW LOOK LIKE:        ##
	##########################################
	##    features =array([[0.65642969],    ##
	##                     [0.64807976],    ##
	##                     [0.65072932],    ##
	##                     ...              ##
	##                     [0.6576711 ]])   ##
	##                                      ##
	##    labels = array([[1., 0., 0.],     ##
	##                    [1., 0., 0.],     ##
	##                    [0., 0., 1.],     ##
	##                    [1., 0., 0.],     ##
	##                    [0., 1., 0.],     ##
	##                    [0., 1., 0.],     ##
	##                    [0., 0., 1.],     ##
	##                    [1., 0., 0.],     ##
	##                    [0., 0., 1.],     ##
	##                    [0., 1., 0.]])    ##
	##                                      ##
	##########################################

	return features, labels 

# skip the CNN for neural architecture search because it doesn't work unless an image type really.
def train_autokeras(classes, alldata, labels, mtype):

	## this is a CNN architecture 
	TEST_FOLDER = "test"
	# features, labels = transform_data(alldata, labels, True)
	# print(features)
	# print(labels)
	x_train, x_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
	x_train=x_train.reshape(x_train.shape + (1,))
	x_test=x_train.reshape(x_train.shape + (1,))

	output_node=2
	input_shape=x_train[0].shape
	print(input_shape)
	n_output_node=(len(classes),1)

	# cnnModule = CnnModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=False)
	if mtype == 'c':
		# metric = Accuracy is for classification 
		# loss = classiciation_loss for classification 
		mlpModule = MlpModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=False)
	elif mtype == 'r':
		# metric = MSE for regression
		# loss = regression_loss for regression
		mlpModule = MlpModule(loss=regression_loss, metric=MSE, searcher_args={}, path=TEST_FOLDER, verbose=False)
	
	print('training MLP model for 1 hour')
	mlpModule.fit(n_output_node, input_shape, x_train, x_test, time_limit=60*60)
	mlpModule.final_fit(x_train, y_train, x_test, y_test, retrain=True)
	y = mlpModule.evaluate(x_test, y_test)
	print(y * 100)
