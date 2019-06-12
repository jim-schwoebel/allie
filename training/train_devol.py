from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from helpers.devol.devol import DEvol, GenomeHandler
from sklearn.model_selection import train_test_split
import time 

def train_devol(classes, alldata, labels, mtype):
	print('training DEVOL CNN network (may take up to 1 day)')
	x_train, x_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# reshape the data (to accomodate library needs)
	x_train=x_train.reshape(x_train.shape+ (1,)+ (1,))
	x_test=x_test.reshape(x_test.shape+ (1,)+ (1,))
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	dataset = ((x_train, y_train), (x_test, y_test))
	# print(dataset)
	# time.sleep(10)
	print(x_train.shape)
	print(x_train[0].shape)
	print(x_test.shape)
	print(x_test[0])
	# time.sleep(10)
	# print(x_train)
	# time.sleep(10)
	# print(y_train.shape)
	# time.sleep(10)
	# print(y_train)

	'''
	The GenomeHandler class handles the constraints that are imposed upon models in a particular genetic program. 
	In this example, a genome is allowed up to 6 convolutional layeres, 3 dense layers, 256 feature maps in each convolution, and 
	1024 nodes in each dense layer. It also specifies three possible activation functions. See genome-handler.py for more information.
	'''

	# prepare genome configuratino 
	genome_handler = GenomeHandler(max_conv_layers=6, 
	                               max_dense_layers=2, # includes final dense layer
	                               max_filters=256,
	                               max_dense_nodes=1024,
	                               input_shape=x_train[0].shape,
	                               n_classes=len(classes))


	'''
	The next, and final, step is create a DEvol and run it. Here we specify a few settings pertaining to the genetic program. 
	In this example, we have 10 generations of evolution, 20 members in each population, and 3 epochs of training used to evaluate 
	each model's fitness. The program will save each genome's encoding, as well as the model's loss and accuracy, in a .csv file printed at the beginning of program.
	'''

	devol = DEvol(genome_handler)
	model = devol.run(dataset=dataset,
	                  num_generations=20,
	                  pop_size=20,
	                  epochs=5)
	model.summary()

