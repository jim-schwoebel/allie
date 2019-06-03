from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np

def vgg19_featurize(file):
    # load model 
    model = VGG19(include_top=True, weights='imagenet')
    img_path = file 
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    # print(features.shape)
    features=np.ndarray.flatten(features)
    # feature shape = (25088,)
    labels=list()
    for i in range(len(features)):
    	labels.append('VGG19_feature_%s'%(str(i+1)))
    return features, labels 


