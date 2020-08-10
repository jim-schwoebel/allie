## Getting started

Here is a way to quickly augment a folder of image files:
```
cd ~ 
cd allie/features/image_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

### Image
* [augment_imaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/image_augmentation/augment_image.py) - uses [imaug](https://github.com/aleju/imgaug) to augment image files (random transformations).

## Settings
| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_image_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation) | the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imaug"] | ["augment_imaug"]  | 
