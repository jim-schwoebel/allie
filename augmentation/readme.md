## Augmentation 

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/augment.png)

This part of Allie's skills relates to data augmentation.

Data augmentation is used to expand the training dataset in order to improve the performance and ability of a machine learning model to generalize. For example, you may want to shift, flip, brightness, and zoom on images to augment datasets to make models perform better in noisy environments indicative of real-world use. Data augmentation is especially useful when you don't have that much data, as it can greatly expand the amount of training data that you have for machine learning. 

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

You can read more about data augmentation [here](https://github.com/AgaMiko/data-augmentation-review).

## Getting started

To augment an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```python3
cd /Users/jim/desktop/allie
cd augmentation/audio_augmentation
python3 augment.py /Users/jim/desktop/allie/train_dir/males/
python3 augment.py /Users/jim/desktop/allie/train_dir/females/
```

The code above will augment all the audio files in the folderpath via the default_augmenter specified in the settings.json file (e.g. 'augment_tasug'). In this case, it will augment both the males and females folders full of .WAV files

You should now have 2x the data in each folder. Here is a sample audio file and augmented audio file (in females) folder, for reference:
* [Non-augmented file sample](https://drive.google.com/file/d/1kvdoKn0IjBXhBEtjDq9AK8CjD14nIC35/view?usp=sharing) (female speaker)
* [Augmented file sample](https://drive.google.com/file/d/1EsSHx1m_zxrdTjnRMhYKOLLjiKi5gRgY/view?usp=sharing) (female speaker)

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/augment.gif)](https://drive.google.com/file/d/1j-rGRCgVDifIzoPx3YNuux_k9H-Gb-TD/view?usp=sharing)

## Expanding to other file types

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
* [augment_tsaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_tsaug.py) - adds noise and various shifts to audio files, addes 2x more data; see tutorial [here]((https://tsaug.readthedocs.io/en/stable/).
* [augment_addnoise](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_addnoise.py) - adds noise to an audio file.
* [augment_noise](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_noise.py) - removes noise from audio files randomly.
* [augment_pitch](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_pitch.py) - shifts pitch up and down to correct for gender differences. 
* [augment_randomsplice](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_randomsplice.py) - randomly splice an audio file to generate more data.
* [augment_silence](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_silence.py) - add silence to an audio file to augment a dataset.
* [augment_time](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_time.py) - change time duration for a variety of audio files through making new files.
* [augment_volume](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_volume.py) - change volume randomly (helps to mitigate effects of microphohne distance on a model).

### [Text](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation)
* [augment_textacy](https://github.com/jim-schwoebel/allie/blob/master/augmentation/text_augmentation/augment_textacy.py) - uses [textacy](https://chartbeat-labs.github.io/textacy/build/html/index.html) to augment text files.

### [Image](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation)
* [augment_imaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/image_augmentation/augment_image.py) - uses [imaug](https://github.com/aleju/imgaug) to augment image files (random transformations).

### [Video](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation)
* [augment_vidaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/video_augmentation/augment_vidaug.py) - uses [vidaug](https://github.com/okankop/vidaug) to augment video files (random transformations).

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

