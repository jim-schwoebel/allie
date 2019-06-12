from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from helpers.devol.devol import DEvol, GenomeHandler

def train_devol():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	K.set_image_data_format("channels_last")

	'''
	This problem uses mnist, a handwritten digit classification problem used for many introductory deep learning examples. Here, we load the data and prepare it for use by the GPU. We also do a one-hot encoding of the labels.
	'''

	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	dataset = ((x_train, y_train), (x_test, y_test))

	'''
	The GenomeHandler class handles the constraints that are imposed upon models in a particular genetic program. In this example, a genome is allowed up to 6 convolutional layeres, 3 dense layers, 256 feature maps in each convolution, and 1024 nodes in each dense layer. It also specifies three possible activation functions. See genome-handler.py for more information.
	'''

	# prepare genome configuratino 
	genome_handler = GenomeHandler(max_conv_layers=6, 
	                               max_dense_layers=2, # includes final dense layer
	                               max_filters=256,
	                               max_dense_nodes=1024,
	                               input_shape=x_train.shape[1:],
	                               n_classes=10)


	'''

	The next, and final, step is create a DEvol and run it. Here we specify a few settings pertaining to the genetic program. In this example, we have 10 generations of evolution, 20 members in each population, and 3 epochs of training used to evaluate each model's fitness. The program will save each genome's encoding, as well as the model's loss and accuracy, in a .csv file printed at the beginning of program.
	'''

	devol = DEvol(genome_handler)
	model = devol.run(dataset=dataset,
	                  num_generations=20,
	                  pop_size=20,
	                  epochs=5)
	model.summary()

train_devol()