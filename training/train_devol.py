from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from helpers.devol.devol import DEvol, GenomeHandler
from sklearn.model_selection import train_test_split
import time, os, shutil 

def train_devol(classes, alldata, labels, mtype, jsonfile, problemtype, default_features):
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
	                  num_generations=1,
	                  pop_size=10,
	                  epochs=10)
	model.summary()

	# get model name 
	modelname=jsonfile[0:-5]+'_devol_%s'%(str(default_features.replace("'",'').replace('"','')))

	score = model.evaluate(x_test, y_test, verbose=0)
	jsonfile=open(modelname+'.json','w')
	data={'accuracy': score[1],
		  'sampletype': mtype,
		  'feature_set': default_features,
		  'model_name': modelname+".h5",
		  'training_type': 'devol',
		  'model summary': str(model.summary()),
		}

	json.dump(data,jsonfile)
	jsonfile.close()

	os.remove('best-model.h5')
	
	# save the model in .h5 format
	model.save(modelname+".h5")
	print("\n Saved %s.json model to disk"%(modelname))

	listdir=os.listdir()
	for i in range(len(listdir)):
		if listdir[i][-4:]=='.csv':
			os.rename(listdir[i], modelname+'.csv')

	cur_dir2=os.getcwd()
	try:
	    os.chdir(problemtype+'_models')
	except:
	    os.mkdir(problemtype+'_models')
	    os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.move(cur_dir2+'/'+modelname+'.json', os.getcwd()+'/'+modelname+'.json')
	shutil.move(cur_dir2+'/'+modelname+'.csv', os.getcwd()+'/'+modelname+'.csv')
	shutil.move(cur_dir2+'/'+modelname+'.h5', os.getcwd()+'/'+modelname+'.h5')

	return modelname+'.h5', os.getcwd()
