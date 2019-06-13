'''
@Train_autokeras script.

Take in a dataset, 
convert it to pytorch dataloader format,
ingest it in autokeras,
output model in './models directory'

This will make it easier to deploy automated machine learning models
into the future. 

Note that grid search can be expensive + take up to 24 hours on most 
GPUs / CPUs to optimize a model.
'''
from autokeras import MlpModule, CnnModule
from autokeras.backend.torch.loss_function import classification_loss
from autokeras.backend.torch.loss_function import regression_loss
from autokeras.nn.metric import Accuracy
from autokeras.utils import pickle_from_file

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
import keras.models
from keras import layers 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
import torch, time, shutil, os 
import torch.utils.data as utils

# skip the CNN for neural architecture search because it doesn't work unless an image type really.
def train_autokeras(classes, alldata, labels, mtype, jsonfile, problemtype, default_features):

	## this is a CNN architecture 
	modelname=jsonfile[0:-5]+'_autokeras_%s'%(default_features)
	TEST_FOLDER = modelname

	x_train, x_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# we have to do some odd re-shapes to get the data loader to work for the autokeras module (keep this in mind when loading new data in)
	x_train=x_train.reshape(x_train.shape+(1,))
	y_train=y_train.reshape(y_train.shape+(1,)+(1,))
	x_test=x_test.reshape(x_test.shape+(1,))
	y_test=y_test.reshape(y_test.shape+(1,)+(1,))
	print(x_train.shape)
	print(y_train.shape)

	tensor_x = torch.stack([torch.Tensor(i) for i in x_train]) # transform to torch tensors
	tensor_y = torch.stack([torch.Tensor(i) for i in y_train])

	my_dataset = utils.TensorDataset(tensor_x, tensor_y) # create your datset
	training_data = utils.DataLoader(my_dataset) # create your dataloader

	tensor_x = torch.stack([torch.Tensor(i) for i in x_test]) # transform to torch tensors
	tensor_y = torch.stack([torch.Tensor(i) for i in y_test])

	my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
	test_data = utils.DataLoader(my_dataset) # create your dataloader
	
	print(test_data)

	input_shape=x_train[0].shape
	n_output_node=1

	# cnnModule = CnnModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=False)
	if mtype == 'c':
		# metric = Accuracy is for classification 
		# loss = classiciation_loss for classification 
		mlpModule = MlpModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=True)
	elif mtype == 'r':
		# metric = MSE for regression
		# loss = regression_loss for regression
		mlpModule = MlpModule(loss=regression_loss, metric=MSE, searcher_args={}, path=TEST_FOLDER, verbose=True)

	timelimit=60
	print('training MLP model for %s hours'%(timelimit/(60*60)))
	mlpModule.fit(n_output_node, input_shape, training_data, test_data, time_limit=timelimit)
	mlpModule.final_fit(training_data, test_data, trainer_args=None, retrain=False)
	
	# # serialize model to JSON
	# mlpModule.export_autokeras_model(modelname+'.pickle')
	# print("\n Saved %s.pickle model to disk"%(modelname))

	# # test opening model and making predictions
	# model=pickle_from_file(modelname+'.pickle')
	# results=model.evaluate(x_test, y_test)
	# print(results)

	cur_dir2=os.getcwd()

	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copytree(cur_dir2+'/'+TEST_FOLDER, os.getcwd() + '/'+TEST_FOLDER)
	shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
	shutil.move(cur_dir2+'/'+modelname+".h5", os.getcwd()+'/'+modelname+".h5")
	shutil.move(cur_dir2+'/'+modelname+".json", os.getcwd()+'/'+modelname+".json")
	shutil.rmtree(cur_dir2+'/'+TEST_FOLDER)
