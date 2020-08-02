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

## active things to finish before a live launch [ongoing list]

### ongoing (for version 1.0.0 release)
- documentation of the repository / jupyter notebooks with examples in research paper 
- re-test allie tests on linux, etc. / contextualize tests around default settings
- solve bug relating to regression problems in the visualize.py script (this does not work for regression)
- solve regression problem loading machine learning models and making predictions (from spreadsheets)
-  {class: {value: value}} prediction / only allow for csv files for training (get regression model prediction working)
- tie new datasets with SurveyLex product / CLI interface with downloads
- Create nice CLI interface for all of Allie's functionality using OptionParser()

### future releases (1.0.1 release)

- enhance visualizers with audio (RMS power/25 samples), text (freqdist plot), image, video, and csv-specific analyses
- add in notion of 'saving' datasets in the ./datasets directory in d3m format, .JSON format, and .CSV file format (and upload these into the cloud on S3 or an FTP server)
- create time_split type of setting (for audio and video files) in annotation
- create live version of annotation script for audio, text, image, and video files (and add-in default_audio_annotators, default_text_annotators, default_image_annotators, and/or default_video_annotators into settings.json)
- clean up datasets folder --> cleaning dir / augmentation dir (these can change to main directory tree), change labeling directory to annotation in main directory
- create single-file annotation mode (instead of folders)
- create single-file prediction mode (instead of folders)
- create single-file featurization mode (instead of folders)
- add single-file cleaning mode (instead of folders)
- add single-file augmentation mode (instead of folders)
- create docker containers for production for any arbitrary data type / specify to AWS, GCP, or Azure deployment (in marketplaces) / Flask with Auth0 integration for custom APIs (submit file --> get back model results)
- add in [statsmodels](https://www.statsmodels.org/stable/index.html) and [MLpy](http://mlpy.sourceforge.net/) dimensionality reduction techniques and modeling techniques
- add in augmentation policies into visualizer to show which augmentation methods work to increase AUC / MSE
- add in cleaning policies into visualizer to show which cleaning methods work to increase AUC / MSE
- add in both cleaning and augmentation policies (in combinatoric fashion) to show which combinations work best for AUC / MSE
- use combinatoric policies to select optimal model from configurations (clean, augmentation, preprocessing techniques, etc.); train_combinatorics.py (new script idea)
- add in new ASR: https://github.com/rolczynski/Automatic-Speech-Recognition

### recently completed (version 1.0.0 release)
- add new test cases into Allie / make tests work with new framework
- fixed loading ML models for video and image types of models
- improved documentation for cleaning and augmentation techniques
- added in text, image, video, and audio cleaning techniques (in new format)
- added in text, image, video, and audio augmentation techniques (in new format)
- add error handling into all of Allie's featurizations + error array into feature array itself ("error" form of column on features)
- kept create_readme setting for making readmes in the repositories themselves (deleted create_YAML setting)
- deleted the production folder schema within Allie
- added component numbers for both dimensionality reducers and feature selectors in settings.json
- fix small bug .JSON files for model files.
- add 'pip3 freeze > requirements.txt' --> to machine learning model training systems to reproduct environments on different CPUs 
- added audio_features/loudness_features.py using pyloudnorm (in dB)
- cleaned up audio_features/sa_feature array to be a simpler # of lines (and made a fixed length-array)
- fixed bug in loading AutoGluon models for making predictions with the load.py script in the ./models/ directory (and loading model_type variable generally)
- add in ['zscore','isolationforest'] to remove outliers (https://stackoverflow.com/questions/51390196/how-to-calculate-cooks-distance-dffits-using-python-statsmodel) - remove_outliers == True / False.
- added a sample validation script in the ./models directory to quickly assess how well machine learning models generalize to new datasets
- added Figlet for cool text renderings / messages when loading modeling scripts (http://www.figlet.org/)
- bug fix - minor bug fix in visualize.py script; fixed loading broken .JSON files during featurization (broke the visualization script during model training)
- bug fix - edited transforms such that they are named by the common name and not by all the classes trained, as if you have >30 classes this will cause the transform to fail at saving / loading
- added option in modeling script to create csv files (if create_csv == True, then creates .CSV files during training) - note the reason for this is for very large files it can take a long time to create them, so model training sessions can be sped up by setting create_csv == False.
- added annotate.py script to annotate files (beta version) - need to add to .JSON schema (in labels (regression)
- come up with the ability to train regression models by a class and value
- add in single model prediction mode in ./load.py script (-audio (sampletype) -c_autokeras (folder) -directory)
- add in all model loaders from the model trainers 
- fixed cvopt and autokaggle training script bugs
- added in the ability to quickly visualize ML models trained in a spreadhseet with the [model2csv.py script](https://github.com/jim-schwoebel/allie/blob/master/models/model2csv.py)
- bug fix - minor bug fixes associated with transcription during featurization for audio, image, video, and .CSV files
- add notion of "tabular" data instead of .CSV to tie to audio, video, and image data (e.g. for loading datasets) - as laid out in the [d3m-schema](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md#case-2) - did this in the [featurize_csv script](https://github.com/jim-schwoebel/allie/blob/master/train_dir/featurize_csv.py) where .CSV files can contain audio, text, image, video, numerical, and categorical data.
- test and validate model compression works for all training scripts / can load compressed models and make predictions (w/ production)
- finish up model trainers and clean them up with standard metrics for accuracy
- add in version to Allie (to assess deprecation issues into the future)
- add in deepspeech functionality to transcription for open source (and other open source audio transcribers)
- add in transcriber settings as a list ['pocketsphinx', 'deepspeech', 'google', 'aws'], etc.
- added in transcribers as lists (can be adapted into future)
- created a version 2 trainer for machine learning models (as part of Allie release 1.0.0)

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

## Settings

Settings can be modified in the settings.json file. If no settings.json file is identified, it will automatically be created with some default settings from the setup.py script.

![](https://github.com/jim-schwoebel/allie/blob/master/datasets/labeling/helpers/Screen%20Shot%202019-09-29%20at%2010.43.36%20PM.png)

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| version | version of Allie release | 1.0 | 1.0 |
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| balance_data | whether or not to balance datasets during the model training process. | True | True, False | 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| create_csv | whether or not to output datasets in a nicely formatted .CSV as part of the model training process (outputs to ./data folder in model repositories) | True | True, False | 
| default_audio_augmenters | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ['normalize_volume', 'normalize_pitch', 'time_stretch', 'opus_enhance', 'trim_silence', 'remove_noise', 'add_noise', "augment_tsaug"] | 
| default_audio_cleaners | the default cleaning strategies used during audio modeling if clean_data == True | ["clean_mono16hz"] | ["clean_getfirst3secs", "clean_keyword", "clean_mono16hz", "clean_mp3towav", "clean_multispeaker", "clean_normalizevolume", "clean_opus", "clean_random20secsplice", "clean_removenoise", "clean_removesilence", "clean_utterances"] |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ['pocketsphinx', 'deepspeech_nodict', 'deepspeech_dict', 'google', 'wit', 'azure', 'bing', 'houndify', 'ibm'] | 
| default_csv_augmenters | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification"]  | 
| default_csv_cleaners| | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv" |
| default_csv_featuers | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| default_dimensionality_reducer | the default dimensionality reduction technique used if reduce_dimensions==True | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| default_feature_selector | the default feature selector used if select_features == True | ["rfe"] | ["lasso", "rfe"]  | 
| default_image_augmenters| the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imaug"] | ["augment_imaug"]  | 
| default_image_cleaners | the default cleaning techniques used for image data as a part of model training is clean_data == True| ["clean_greyscale"] |["clean_extractfaces", "clean_greyscale", "clean_jpg2png"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| default_outlier_detector | the default outlier technique(s) used to clean data as a part of model training if remove_outliers == True | ["isolationforest"] | ["isolationforest","zscore"]  | 
| default_scaler | the default scaling technique used to preprocess data during model training if scale_features == True | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"] | 
| default_text_augmenters | the default augmentation strategies used during model training for text data if augment_data == True | ["augment_textacy"] | ["augment_textacy", "augment_summary"]  | 
| default_text_cleaners | the default cleaning techniques used during model training on text data if clean_data == True| ["clean_textacy"] | ["clean_textacy"]  | 
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| default_text_transcriber | the default transcription techniques used to parse raw .TXT files during model training| ["raw_text"] | ["raw_text"]  | 
| default_training_script | the specified traning script(s) to train machine learning models. Note that if you specify multiple training scripts here that the training scripts will be executed serially (list). | ['tpot'] |['alphapy', 'atm', 'autogbt', 'autokaggle', 'autokeras', 'auto-pytorch', 'btb', 'cvopt', 'devol', 'gama', 'hyperband', 'hypsklearn', 'hungabunga', 'imbalance-learn', 'keras', 'ludwig', 'mlblocks', 'neuraxle', 'safe', 'scsr', 'tpot']| 
| default_video_augmenters | the default augmentation strategies used for videos during model training if augment_data == True | ["augment_vidaug"] | ["augment_vidaug"] | 
| default_video_cleaners | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
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

## License
This repository is licensed under an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). 

## Feedback
Any feedback on the book or this repository is greatly appreciated. 
* If you find something that is missing or doesn't work, please consider opening a [GitHub issue](https://github.com/jim-schwoebel/Allie/issues).
* If you'd like to be mentored by someone on our team, check out the [Innovation Fellows Program](http://neurolex.ai/research).
* If you want to talk to me directly, please send me an email @ js@neurolex.co. 

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
