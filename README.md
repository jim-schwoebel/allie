# Allie 

Allie is a framework for building machine learning models from audio, text, image, video, or .CSV files.

Here are some things that Allie can do:
- featurize files and export data in .CSV format (via audio, text, image, video, or csv featurizers)
- transform features (via NNI, scikit-learn, and transformers)
- create visualizations from featurized datasets (via yellowbrick, scikit-learn, and matplotlib libraries)
- train machine learning models (via tpot, hyperopt, scsr, devol, keras, ludwig training scripts)
- make predictions from machine learning models (with all models trained in ./models directory)
- prepare compressed machine learning models for deployment (including repositories with readmes)

![](https://media.giphy.com/media/20NLMBm0BkUOwNljwv/giphy.gif)

You can read more about Allie in the [wiki documentation](https://github.com/jim-schwoebel/allie/wiki).

## active things to finish before a live launch [ongoing list]

### ongoing 

1. add in default_augmenters / get live into Allie
2. add in default_cleaners / get live into Allie 
5. add in all model loaders from the model trainers 
6. test and validate model compression works for all training scripts / can load compressed models and make predictions (w/ production)
7. create docker containers for production for any arbitrary data type 
8. add notion of "tabular" data instead of .CSV to tie to audio, video, and image data (e.g. for loading datasets) - as laid out in the [d3m-schema](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md#case-2)
9. tie new datasets with SurveyLex product / CLI interface with downloads
10. documentation of the repository / jupyter notebooks with examples in research paper 
11. Create nice CLI interface for all of Allie's functionality using OptionParser()
12. make sure Allie passes all tests on linux, etc. / contextualize tests around default settings
13. Add in [statsmodels](https://www.statsmodels.org/stable/index.html) and [MLpy](http://mlpy.sourceforge.net/) dimensionality reduction techniques and modeling techniques 

### recently completed
- finish up model trainers and clean them up
- add in version to Allie (to assess deprecation issues into the future)
- add in deepspeech functionality to transcription for open source (and other open source audio transcribers)
- add in transcriber settings as a list ['pocketsphinx', 'deepspeech', 'google', 'aws'], etc.
- added in transcribers as lists (can be adapted into future)

## getting started 

### MacOS
First, clone the repository:
```
git clone --recurse-submodules -j8 git@github.com:jim-schwoebel/allie.git
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
Note the unit tests above takes roughly 30-40 minutes to complete and makes sure that you can featurize, model, and load model files (to make predictions) via your default featurizers and modeling techniques. It may be best to go grab lunch or coffee while waiting. :-)

### Windows 10

#### recommended installation (Docker)

You can run Allie in a Docker container fairly easily (10-11GB container run on top of Linux/Ubuntu):

```
git clone --recurse-submodules -j8 git@github.com:jim-schwoebel/allie.git
cd allie 
docker build -t allie_image .
docker run -it --entrypoint=/bin/bash allie_image
cd ..
```

You will then have access to the docker container to use Allie's folder structure. You can then run tests @

```
cd tests
python3 test.py
```

#### alternative

Note that there are many incomptible Python libraries with Windows, so I encourage you to instead run Allie in a Docker container with Ubuntu or on [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

If you still want to try to use Allie with Windows, you can do so below. 

First, install various dependencies:

- Download Microsoft Visual C++ (https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15).
- Download SWIG and compile locally as an environment variable (http://www.swig.org/download.html).
- Follow instructions to setup [Tensorflow](https://medium.com/@amsokol.com/how-to-build-and-install-tensorflow-gpu-cpu-for-windows-from-source-code-using-bazel-d047d9342b44) on Windows.

Now clone Allie and run the setup.py script:
```
git clone --recurse-submodules -j8 git@github.com:jim-schwoebel/allie.git
git checkout windows
cd allie 
python3 -m pip install --user virtualenv
python3 -m venv env
python3 setup.py
```

Note that there are some functions that are limited (e.g. featurization / modeling scripts) due to lack of Windows compatibility.

### Linux

Here are the instructions for setting up Allie on Linux:

```
git clone --recurse-submodules -j8 git@github.com:jim-schwoebel/allie.git
git checkout linux
cd allie 
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
python3 setup.py
```

Now you can run some unit tests:
```
cd tests
python3 test.py
```

## folder structures

Here is a table that describes the folder structure for this repository. These descriptions could help guide how you can quickly get started with featurizing and modeling data samples. 

| folder name | description of folder |
| ------- | ------- |
| [datasets](https://github.com/jim-schwoebel/Allie/tree/master/datasets) | an elaborate list of open source datasets that can be used for curating, cleaning, and augmenting datasets. |
| [features](https://github.com/jim-schwoebel/Allie/tree/master/features) | a list of audio, text, image, video, and csv featurization scripts (defaults can be specified in the settings.json files). |
| [load_dir](https://github.com/jim-schwoebel/Allie/tree/master/load_dir) | a directory where you can put in audio, text, image, video, or .CSV files and make model predictions from ./models directory. | 
| [models](https://github.com/jim-schwoebel/Allie/tree/master/training) | for loading/storing machine learning models and making model predictions for files put in the load_dir. | 
| [production](https://github.com/jim-schwoebel/Allie/tree/master/production) | a folder for outputting production-ready repositories via the YAML.py script. | 
| [tests](https://github.com/jim-schwoebel/Allie/tree/master/tests) | for running local tests and making sure everything works as expected. | 
| [train_dir](https://github.com/jim-schwoebel/Allie/tree/master/train_dir) | a directory where you can put in audio, text, image, video, or .CSV files in folders and train machine learning models from the model.py script in the ./training/ directory. |
| [training](https://github.com/jim-schwoebel/Allie/tree/master/models) | for training machine learning models via specified model training scripts. |
| [visualize](https://github.com/jim-schwoebel/Allie/tree/master/visualize) | for visualizing and selecting features as part of the model creation process. |

## standard feature array 

After much trial and error, this standard feature array schema seemed the most appropriate for defining data samples (audio, text, image, video, or CSV samples):

```python3
def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		  'text': dict(),
		  'image':dict(),
		  'video':dict(),
		  'csv': dict()}

	transcripts={'audio': dict(),
		     'text': dict(),
		     'image': dict(),
		     'video': dict(),
		     'csv': dict()}

	models={'audio': dict(),
		 'text': dict(),
		 'image': dict(),
		 'video': dict(),
		 'csv': dict()}

	data={'sampletype': sampletype,
	      'transcripts': transcripts,
	      'features': features,
	      'models': models,
	      'labels': []}

	return data
```

There are many advantages for having this schema including:
- **sampletype definition flexibility** - flexible to 'audio' (.WAV / .MP3), 'text' (.TXT / .PPT / .DOCX), 'image' (.PNG / .JPG), 'video' (.MP4), and 'csv' (.CSV). This format can also can adapt into the future to new sample types, which can also tie to new featurization scripts. By defining a sample type, it can help guide how data flows through model training and prediction scripts. 
- **transcript definition flexibility** - transcripts can be audio, text, image, video, and csv transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). You can also add multiple transcripts (e.g. Google and PocketSphinx) for the same sample type.
- **featurization flexibility** - many types of features can be put into this array of the same data type. For example, an audio file can be featurized with 'standard_features' and 'praat_features' without really affecting anything. This eliminates the need to re-featurize and reduces time to sort through multiple types of featurizations during the data cleaning process.
- **label annotation flexibility** - can take the form of ['classname_1', 'classname_2', 'classname_N...'] - classification problems and [{classname1: 'value'}, {classname2: 'value'}, ... {classnameN: 'valueN'}] where values are between [0,1] for regression problems. 
- **model predictions** - one survey schema can be used for making model predictions and updating the schema with these predictions. Note that any model that is used for training can be used to make predictions in the load_dir. 
- **visualization flexibility** - can easily visualize features of any sample tpye through Allie's [visualization script](https://github.com/jim-schwoebel/allie/tree/master/visualize) (e.g. tSNE plots, correlation matrices, and more).

This schema is inspired by [D3M-schema](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md) by the MIT media lab.

We are currently in process to implement this schema into the SurveyLex architecture. 

## easy data exports

Easily featurize and export data in .CSV format for porting data across ML platforms. This is useful for benchmarking and curating datasets that are repeatable.

```
Show example of this here
```

## settings 

Settings can be modified in the settings.json file. If no settings.json file is identified, it will automatically be created with some default settings from the setup.py script.

![](https://github.com/jim-schwoebel/allie/blob/master/datasets/labeling/helpers/Screen%20Shot%202019-09-29%20at%2010.43.36%20PM.png)

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "standard_features"] | 
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| [default_csv_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of csv features used for featurization (list). | ["csv_features"] | ["csv_features"] | 
| transcribe_audio | determines whether or not to transcribe an audio file via default_audio_transcriber (boolean). | True | True, False | 
| default_audio_transcriber | the default audio transcriber if transcribe_audio == True (list). | ['pocketsphinx'] | ['pocketsphinx', 'deepspeech_nodict', 'deepspeech_dict', 'google', 'wit', 'azure', 'bing', 'houndify', 'ibm'] | 
| transcribe_text | determines whether or not to transcribe a text file via default_text_transcriber (boolean). | True | True, False | 
| default_text_transcriber | the default text transcriber if transcribe_text == True (list). | ['raw text'] | ['raw text'] | 
| transcribe_image | determines whether or not to transcribe an image file via default_image_transcriber (boolean). | True | True, False | 
| default_image_transcriber | the default image transcriber if transcribe_image == True (list). | ['tesseract'] | ['tesseract'] | 
| transcribe_video | determines whether or not to transcribe a video file via default_video_transcriber (boolean). | True | True, False | 
| default_video_transcriber | the default video transcriber if transcribe_video == True (boolean). | ['tesseract_connected_over_frames'] | ['tesseract_connected_over_frames'] | 
| transcribe_csv | determines whether or not to transcribe a csv file via default_csv_transcriber (boolean). | True | True, False | 
| default_csv_transcriber | the default video transcriber if transcribe_csv == True (list). | ['raw text'] | ['raw text'] | 
| default_training_script | the specified traning script(s) to train machine learning models. Note that if you specify multiple training scripts here that the training scripts will be executed serially (list). | ['tpot'] |['alphapy', 'atm', 'autogbt', 'autokaggle', 'autokeras', 'auto-pytorch', 'btb', 'cvopt', 'devol', 'gama', 'hyperband', 'hypsklearn', 'hungabunga', 'imbalance-learn', 'keras', 'ludwig', 'mlblocks', 'neuraxle', 'safe', 'scsr', 'tpot']| 
| clean_data | specifies whether or not you'd like to clean / pre-process data in folders before model training (boolean). |  True | True, False | 
| default_audio_cleaners | the specified cleaning scripts to employ when cleaning audio data | ['remove_duplicates'] | ['remove_duplicates'] |
| default_text_cleaners | the specified cleaning scripts to employ when cleaning text data | ['remove_duplicates'] | ['remove_duplicates'] |
| default_image_cleaners | the specified cleaning scripts to employ when cleaning image data | ['remove_duplicates'] | ['remove_duplicates'] |
| default_video_cleaners | the specified cleaning scripts to employ when cleaning video data | ['remove_duplicates'] | ['remove_duplicates'] |
| default_csv_cleaners | the specified cleaning scripts to employ when cleaning csv data | ['remove_duplicates'] | ['remove_duplicates'] |
| augment_data | specifies whether or not you'd like to augment data during training (boolean). |  False | True, False | 
| default_audio_augmenters | the specified cleaning scripts to employ when augmenting audio data | ['normalize_volume', 'add_noise', 'time_stretch'] | ['normalize_volume', 'normalize_pitch', 'time_stretch', 'opus_enhance', 'trim_silence', 'remove_noise', 'add_noise'] |
| default_text_augmenters | the specified cleaning scripts to employ when augmenting text data | [] | [] |
| default_image_augmenters | the specified cleaning scripts to employ when augmenting image data | [] | [] |
| default_video_augmenters | the specified cleaning scripts to employ when augmenting video data | [] | [] |
| default_csv_augmenters | the specified cleaning scripts to employ when augmenting csv data | [] | [] |
| reduce_dimensions | if True, reduce dimensions via the default_dimensionality_reducer (or set of dimensionality reducers) | False | True, False |
| default_dimensionality_reducer | the default dimensionality reducer or set of dimensionality reducers | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| select_features | if True, select features via the default_feature_selector (or set of feature selectors) | False | True, False | 
| default_feature_selector | the default feature selector or set of feature selectors | ["lasso"] | ["lasso", "rfe"] | 
| scale_features | if True, scales features via the default_scaler (or set of scalers) | False | True, False | 
| default_scaler | the default scaler (e.g. StandardScalar) to pre-process data | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"]|
| create_YAML | specifies whether or not you'd like to output a production-ready repository for model deployment (boolean). |  False | True, False | 
| model_compress | if True compresses the model for production purposes to reduce memory consumption. Note this only can happen on Keras or scikit-learn / TPOT models for now (boolean).| False | True, False | 

## License
This repository is licensed under a trade secret. Please do not share this code outside the core team.

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
