'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  

______         _                          ___  ______ _____     
|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
 _____                           
|_   _|                          
  | | _ __ ___   __ _  __ _  ___ 
  | || '_ ` _ \ / _` |/ _` |/ _ \
 _| || | | | | | (_| | (_| |  __/
 \___/_| |_| |_|\__,_|\__, |\___|
                       __/ |     
                      |___/      
                      
Extracts image features if default_image_features = ['squeezenet_features']

The Squeezenet model reduces featurization time for large featurizations, 
as it is a compressed model that uses less memory.

Read more about the SqueezeNet model @ https://github.com/rcmalli/keras-squeezenet
'''
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
