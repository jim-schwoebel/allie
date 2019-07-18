import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


def squeezenet_featurize(imagename, imagedir):
	''' 
	This network model has AlexNet accuracy with small footprint (5.1 MB) 
	Pretrained models are converted from original Caffe network.

	This may be useful for production-purposes if the accuracy is similar to other
	types of featurizations.

	See https://github.com/rcmalli/keras-squeezenet
	'''

	model = SqueezeNet()

	img = image.load_img(imagedir+'/'+imagename, target_size=(227, 227))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	print('Predicted:', decode_predictions(preds))
	features = preds[0]
	labels=list()

	for i in range(len(features)):
		label='squeezenet_feature_%s'%(str(i))
		labels.append(label)

	return features, labels 
