# Allie 
Allie is a framework for building machine learning models from audio, text, image, video, or .CSV files.

Here are some things that Allie can do:
- [find](https://github.com/jim-schwoebel/allie/tree/master/datasets) and download datasets (for quick experiments)
- [annotate](https://github.com/jim-schwoebel/allie/tree/master/annotation), [clean](https://github.com/jim-schwoebel/allie/tree/master/cleaning), and/or [augment](https://github.com/jim-schwoebel/allie/tree/master/augmentation) audio, text, image, or video datasets (to prepare data for modeling)
- [featurize](https://github.com/jim-schwoebel/allie/tree/master/features) files using a standard format (via audio, text, image, video, or csv featurizers)
- [transform](https://github.com/jim-schwoebel/allie/tree/master/preprocessing) features (via scikit-learn preprocessing techniques)
- [visualize](https://github.com/jim-schwoebel/allie/tree/master/visualize) featurized datasets (via yellowbrick, scikit-learn, and matplotlib libraries)
- [train](https://github.com/jim-schwoebel/allie/tree/master/training) classification or regression machine learning models (via tpot, hyperopt, scsr, devol, keras, ludwig, and 15 other training scripts)
- [make predictions](https://github.com/jim-schwoebel/allie/tree/master/load_dir) from machine learning models (with all models trained in ./models directory)
- [export data](https://github.com/jim-schwoebel/allie/tree/master/training) in .CSV file formats (for repeatable machine learning experiments across frameworks)
- [compress](https://github.com/jim-schwoebel/allie/tree/master/training) machine learning models for deployment (including repositories with readmes)

![](https://media.giphy.com/media/20NLMBm0BkUOwNljwv/giphy.gif)

You can read more about Allie in the [wiki documentation](https://github.com/jim-schwoebel/allie/wiki).

## getting started (Mac or Linux) 

First, clone the repository:
```
git clone git@github.com:jim-schwoebel/allie.git
cd allie 
```
Set up virtual environment (to ensure consistent operating mode across operating systems).
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
```
Now install required dependencies and perform unit tests to make sure everything works:
```
python3 setup.py
```

Now you can run some unit tests:
```
cd tests
python3 test.py
```
Note the unit tests above takes roughly ~10 minutes to complete and makes sure that you can featurize, model, and load model files (to make predictions) via your default featurizers and modeling techniques. It may be best to go grab lunch or coffee while waiting. :-)

## Quick examples

### [Collecting data](https://github.com/jim-schwoebel/allie/tree/master/augmentation)

To illustrate a quick example, we can pull some sample audio data from this GitHub repository, separating males (x50) from females (x50). 

This [seed_test.py script](https://github.com/jim-schwoebel/allie/blob/master/datasets/seed_test.py) creates two datasets in the train_dir folder, one full of audio files of males and the other full of audio files of females. This data will be used for the rest of the demo sections listed here.

```python3
cd /Users/jim/desktop/allie
cd datasets
python3 seed_test.py
---------------
Cloning into 'sample_voice_data'...
remote: Enumerating objects: 119, done.
remote: Counting objects: 100% (119/119), done.
remote: Compressing objects: 100% (115/115), done.
remote: Total 119 (delta 5), reused 108 (delta 2), pack-reused 0
Receiving objects: 100% (119/119), 18.83 MiB | 7.43 MiB/s, done.
Resolving deltas: 100% (5/5), done.
```
You can easily test if the files are in there with:
```
cd ..
cd train_dir
ls
```
Which should output:
```
jim@Jims-MBP train_dir % ls
README.md		delete_json.py		females
delete_features.py	featurize_csv.py	males
```

### [Annotating data](https://github.com/jim-schwoebel/allie/tree/master/annotation)

You can simply annotate using the command-line interface here:

```python3
cd /Users/jim/desktop/allie
cd annotation
python3 annotate.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

After you annotate, you can create [a nicely formatted .CSV](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/male_data.csv) for machine learning:

```python3
cd /Users/jim/desktop/allie
python3 create_csv.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

Click the .GIF below for a quick tutorial and example.

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/annotation.gif)](https://drive.google.com/file/d/1Xn7A61XWY8oCAfMmjSMpwEjvItiNp5ev/view?usp=sharing)

### [Augmenting data](https://github.com/jim-schwoebel/allie/tree/master/augmentation)

```python3
cd /Users/jim/desktop/allie
cd augmentation/audio_augmentation
python3 augment.py /Users/jim/desktop/allie/train_dir/males/
python3 augment.py /Users/jim/desktop/allie/train_dir/females/
```

You should now have 2x the data in each folder. Here is a sample audio file and augmented audio file (in females) folder, for reference:
* [Non-augmented file](https://drive.google.com/file/d/1kvdoKn0IjBXhBEtjDq9AK8CjD14nIC35/view?usp=sharing)
* [Augmented file](https://drive.google.com/file/d/1EsSHx1m_zxrdTjnRMhYKOLLjiKi5gRgY/view?usp=sharing)

### [Cleaning data](https://github.com/jim-schwoebel/allie/tree/master/cleaning)
Docs here
```python3
cd /Users/jim/desktop/allie
cd cleaning/audio_cleaning
python3 clean.py /Users/jim/desktop/allie/train_dir/males/
```

### [Featurizing data](https://github.com/jim-schwoebel/allie/tree/master/features)

```python3
cd /Users/jim/desktop/allie
cd features/audio_features
python3 featurize.py /Users/jim/desktop/allie/train_dir/males/
```

### [Transforming data](https://github.com/jim-schwoebel/allie/tree/master/preprocessing)

```python3
cd /Users/jim/desktop/allie
cd preprocessing
python3 transform.py /Users/jim/desktop/allie/train_dir/males/
```

### [Modeling data](https://github.com/jim-schwoebel/allie/tree/master/training)
#### classification problem

```python3
cd /Users/jim/desktop/allie
cd training
python3 model.py
```

#### regression problem 
Regression problems require a .CSV spreadsheet with annotations. This can be done with the [Allie's annotate API](https://github.com/jim-schwoebel/allie/blob/master/annotation/annotate.py).

```python3
cd /Users/jim/desktop/allie
cd training
python3 model.py
```

### [Visualizing data](https://github.com/jim-schwoebel/allie/tree/master/visualize)
```python3
cd /Users/jim/desktop/allie
cd visualize
python3 visualize.py audio males females
```

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/settings.png)

Allie has multiple default settings for model training to help you start out with the framework. Allie has been built so that the settings you specify in lists are done serially, which can be useful to construct machine learning models from multiple back-end model trainers in a single session. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| version | version of Allie release | 1.0 | 1.0 |
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| balance_data | whether or not to balance datasets during the model training process. | True | True, False | 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| create_csv | whether or not to output datasets in a nicely formatted .CSV as part of the model training process (outputs to ./data folder in model repositories) | True | True, False | 
| [default_audio_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation) | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ['normalize_volume', 'normalize_pitch', 'time_stretch', 'opus_enhance', 'trim_silence', 'remove_noise', 'add_noise', "augment_tsaug"] | 
| [default_audio_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/audio_cleaning) | the default cleaning strategies used during audio modeling if clean_data == True | ["clean_mono16hz"] | ["clean_getfirst3secs", "clean_keyword", "clean_mono16hz", "clean_mp3towav", "clean_multispeaker", "clean_normalizevolume", "clean_opus", "clean_random20secsplice", "clean_removenoise", "clean_removesilence", "clean_utterances"] |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| [default_csv_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation) | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification", "augment_ctgan_regression"]  | 
| [default_csv_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning) | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv"] | 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| [default_dimensionality_reducer](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py) | the default dimensionality reduction technique used if reduce_dimensions==True | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| [default_feature_selector](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py) | the default feature selector used if select_features == True | ["rfe"] | ["chi", "fdr", "fpr", "fwe", "lasso", "percentile", "rfe", "univariate", "variance"]  | 
| [default_image_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation) | the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imaug"] | ["augment_imaug"]  | 
| [default_image_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning) | the default cleaning techniques used for image data as a part of model training is clean_data == True| ["clean_greyscale"] |["clean_extractfaces", "clean_greyscale", "clean_jpg2png"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| default_outlier_detector | the default outlier technique(s) used to clean data as a part of model training if remove_outliers == True | ["isolationforest"] | ["isolationforest", "zscore"]  | 
| [default_scaler](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_scale.py) | the default scaling technique used to preprocess data during model training if scale_features == True | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"] | 
| [default_text_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation) | the default augmentation strategies used during model training for text data if augment_data == True | ["augment_textacy"] | ["augment_textacy", "augment_summary"]  | 
| [default_text_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/text_cleaning) | the default cleaning techniques used during model training on text data if clean_data == True| ["clean_textacy"] | ["clean_summary", "clean_textacy"]  | 
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| default_text_transcriber | the default transcription techniques used to parse raw .TXT files during model training| ["raw_text"] | ["raw_text"]  | 
| [default_training_script](https://github.com/jim-schwoebel/allie/tree/master/training) | the specified traning script(s) to train machine learning models. Note that if you specify multiple training scripts here that the training scripts will be executed serially (list). | ["tpot"] |["alphapy", "atm", "autogbt", "autokaggle", "autokeras", "auto-pytorch", "btb", "cvopt", "devol", "gama", "hyperband", "hypsklearn", "hungabunga", "imbalance-learn", "keras", "ludwig", "mlblocks", "neuraxle", "safe", "scsr", "tpot"]| 
| [default_video_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation) | the default augmentation strategies used for videos during model training if augment_data == True | ["augment_vidaug"] | ["augment_vidaug"] | 
| [default_video_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning) | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| default_video_transcriber | the default transcription technique used for videos (.mp4 --> text from the video) | ["tesseract (averaged over frames)"] | ["tesseract (averaged over frames)"] |
| dimension_number | the number of dimensions to reduce a dataset into if reduce_dimensions == True| 100 | any integer from 1 to the number of features-1 | 
| feature_number | the number of features to select for via the feature selection strategy (default_feature_selector) if select_features == True| 20 | any integer from 1 to the number of features-1 | 
| model_compress | a setting that specifies whether or not to compress machine learning models during model training | False | True, False | 
| reduce_dimensions | a setting that specifies whether or not to reduce dimensions via the default_dimensionality_reducer | False | True, False | 
| remove_outliers | a setting that specifies whether or not to remove outliers during model training via the default_outlier_detector | True | True, False | 
| scale_features | a setting that specifies whether or not to scale features during featurization and model training via the default_scaler | True | True, False | 
| select_features | a setting that specifies whether or not to employ specified feature selection strategies (via the default_feature_selector) | True | True, False | 
| test_size | a setting that specifies the size of the testing dataset for defining model performance after model training. | 0.10 | Any number 0.10-0.50 | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
| transcribe_csv | a setting to define whether or not to transcribe csv files during featurization and model training via the default_csv_transcriber | True | True, False | 
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_text | a setting to define whether or not to transcribe text files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_video | a setting to define whether or not to transcribe video files during featurization and model training via the default_video_transcriber | True | True, False | 
| [visualize_data](https://github.com/jim-schwoebel/allie/tree/master/visualize) | a setting to define whether or not to visualize features during the model training process via [Allie's visualization API](https://github.com/jim-schwoebel/allie/tree/master/visualize) | False | True, False | 

## License
This repository is licensed under an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). 

## Feedback
Any feedback on this repository is greatly appreciated. We have built this machine learning framework to be quite agile to fit many purposes and needs, and we're excited to see how the open source community uses it. Forks and PRs are encouraged!
* If you find something that is missing or doesn't work, please consider opening a [GitHub issue](https://github.com/jim-schwoebel/Allie/issues).
* If you'd like to be mentored by someone on our team, check out the [Innovation Fellows Program](http://neurolex.ai/research).
* If you want to be a core contributer, please send me an email @ js@neurolex.co. 

## Additional resources

You may want to read through [the wiki](https://github.com/jim-schwoebel/allie/wiki) for additional documentation.

* [0. Getting started](https://github.com/jim-schwoebel/allie/wiki/0.-Getting-started)
* [1. Sample schema](https://github.com/jim-schwoebel/allie/wiki/1.-Sample-schema)
* [2. Collecting datasets](https://github.com/jim-schwoebel/allie/wiki/2.-Collecting-datasets)
* [3. Cleaning datasets](https://github.com/jim-schwoebel/allie/wiki/3.-Cleaning-datasets)
* [4. Augmenting datasets](https://github.com/jim-schwoebel/allie/wiki/4.-Augmenting-datasets)
* [5. Labeling datasets](https://github.com/jim-schwoebel/allie/wiki/5.-Labeling-datasets)
* [6. Data featurization](https://github.com/jim-schwoebel/allie/wiki/6.-Data-featurization)
* [7. Training models](https://github.com/jim-schwoebel/allie/wiki/7.-Training-models)
* [8. Loading models](https://github.com/jim-schwoebel/allie/wiki/8.-Loading-models)
* [9. Server deployment](https://github.com/jim-schwoebel/allie/wiki/9.-Server-deployment)
