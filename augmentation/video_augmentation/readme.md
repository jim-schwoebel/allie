## Getting started

Here is a way to quickly augment a folder of video files:
```
cd ~ 
cd allie/features/video_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

### Video
* [augment_vidaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/video_augmentation/augment_vidaug.py) - uses [vidaug](https://github.com/okankop/vidaug) to augment video files (random transformations).

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_video_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation) | the default augmentation strategies used for videos during model training if augment_data == True | ["augment_vidaug"] | ["augment_vidaug"] | 
