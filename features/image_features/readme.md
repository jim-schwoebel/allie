## How to use

```
cd allie/features/image_features
python3 featurize.py [folder] [featuretype]
```

### Images 
* [image_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/image_features.py) - standard image feature array (default).
* [inception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/inception_features.py) - features extracted with the [Inception model](https://keras.io/api/applications/inceptionv3/).
* [resnet_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/resnet_features.py) - features extracted with the [ResNet model](https://keras.io/api/applications/resnet/#resnet50v2-function).
* [squeezenet_features](https://github.com/rcmalli/keras-squeezenet) - features extracted with the [Squeezenet model](https://github.com/forresti/SqueezeNet); this has an efficient memory footprint.
* [tesseract_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/tesseract_features.py) - features extracted with OCR on images using the [pytesseract module](https://pypi.org/project/pytesseract/).
* [vgg16_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg16_features.py) - features extracted with hte [VGG16 model](https://keras.io/api/applications/vgg/#vgg16-function).
* [vgg19_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg19_features.py) - features extracted with hte [VGG19 model](https://keras.io/api/applications/vgg/#vgg19-function).
* [xception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/xception_features.py) - features extracted with hte [Xception model](https://keras.io/api/applications/xception/).

### Settings
| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
