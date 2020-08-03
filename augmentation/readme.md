## Augmentation 

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/augment.png)

This part of Allie's skills relates to data augmentation.

Data augmentation is used to expand the training dataset in order to improve the performance and ability of a machine learning model to generalize. For example, you may want to shift, flip, brightness, and zoom on images to augment datasets to make models perform better in noisy environments indicative of real-world use. Data augmentation is especially useful when you don't have that much data, as it can greatly expand the amount of training data that you have for machine learning. 

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

You can read more about data augmentation [here](https://github.com/AgaMiko/data-augmentation-review).

## Getting started

To augment an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```
cd ~ 
cd allie/features/audio_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

The code above will augment all the audio files in the folderpath via the default_augmenter specified in the settings.json file (e.g. 'augment_tasug'). 

Note you can extend this to any of the augmentation types. The table below overviews how you could call each as a augmenter. In the code below, you must be in the proper folder (e.g. ./allie/augmentation/audio_augmentations for audio files, ./allie/augmentation/image_augmentation for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 augment.py [folderpath]``` | ./allie/augmentation/audio_augmentation| 
| text files | .TXT | ```python3 augment.py [folderpath]``` | ./allie/augmentation/text_augmentation| 
| image files | .PNG | ```python3 augment.py [folderpath]``` | ./allie/augmentation/image_augmentation | 
| video files | .MP4 | ```python3 augment.py [folderpath]``` |./allie/augmentation/video_augmentation| 
| csv files | .CSV | ```python3 augment.py [folderpath]``` | ./allie/augmentation/csv_augmentation | 

## Implemented 

![](https://github.com/AgaMiko/data-augmentation-review/raw/master/images/da_diagram_v2.png)

### [Audio](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation)
* [augment_tsaug](https://tsaug.readthedocs.io/en/stable/) - see tutorial here.
* [augment_addnoise]()
* [augment_noise]()
* [augment_pitch]()
* [augment_randomsplice]()
* [augment_silence]()
* [augment_time]()
* [augment_volume]()

### [Text](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation)
* [augment_textacy]()

### [Image](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation)
* [augment_imaug]()

### [Video](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation)
* [augment_vidaug]()

### [CSV](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation)
* [augment_tgan_classification](https://github.com/sdv-dev/TGAN) - generative adverserial examples - can be done on class targets / problems.
* [augment_ctgan_regression]() - generative adverserial example on regression problems / targets.

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

Here are some settings that can be customized for Allie's augmentation API. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 

| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_audio_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation) | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ['normalize_volume', 'normalize_pitch', 'time_stretch', 'opus_enhance', 'trim_silence', 'remove_noise', 'add_noise', "augment_tsaug"] | 
| [default_csv_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation) | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification", "augment_ctgan_regression"]  | 
| [default_image_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation) | the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imaug"] | ["augment_imaug"]  | 
| [default_text_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation) | the default augmentation strategies used during model training for text data if augment_data == True | ["augment_textacy"] | ["augment_textacy", "augment_summary"]  | 
| [default_video_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation) | the default augmentation strategies used for videos during model training if augment_data == True | ["augment_vidaug"] | ["augment_vidaug"] | 

## References
* [Review of data augmentation strategies](https://github.com/AgaMiko/data-augmentation-reviewc)
* [1000x Faster Data Augmentation](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c)

