## How to use

```
cd allie/features/image_features
python3 featurize.py [folder] [featuretype]
```

### Images 
* [image_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/image_features.py) - standard image feature array (default)
* [inception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/inception_features.py)
* [resnet_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/resnet_features.py)
* [squeezenet_features](https://github.com/rcmalli/keras-squeezenet) - efficient memory footprint
* [tesseract_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/tesseract_features.py)
* [vgg16_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg16_features.py) 
* [vgg19_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg19_features.py) 
* [xception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/xception_features.py) 

### Settings
| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
