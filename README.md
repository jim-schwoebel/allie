# Allie 

Allie is a framework for building machine learning models from audio, text, image, video, or .CSV files. This is an upgrade from the prior archived [NLX-model repository](https://github.com/NeuroLexDiagnostics/nlx-model).

Here are some things that Allie can do:
- featurize fles (via audio, text, image, video, or csv featurizers)
- train machine learning models (via tpot, hyperopt, scsr, devol, keras training scripts)
- make predictions from machine learning models (with all models trained in models directory)
- prepare machine learning models for deployment (including repositories with readmes)

![](https://media.giphy.com/media/20NLMBm0BkUOwNljwv/giphy.gif)

## getting started 

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
Now install required dependencies:
```
python3 setup.py
```
Now do some unit tests to make sure everything works:
```
python3 tests/test.py
```
Note the test above takes roughly 5-10 minutes to complete and makes sure that you can featurize, model, and load model files (to make predictions) via your default featurizers and modeling techniques.

## folder structures

Here is a table with all the folders and what they are intended to be used for. 

| folder name | description of folder |
| ------- | ------- |
| [datasets](https://github.com/jim-schwoebel/Allie/tree/master/datasets) | an elaborate list of open source datasets that can be used for curating datasets and augmenting datasets. |
| [features](https://github.com/jim-schwoebel/Allie/tree/master/features) | a list of audio, text, image, video, and csv featurization scripts (these can be specified in the settings.json files). |
| [load_dir](https://github.com/jim-schwoebel/Allie/tree/master/load_dir) | a directory where you can put in audio, text, image, video, or .CSV files and make moel predictions from ./models directory. | 
| [models](https://github.com/jim-schwoebel/Allie/tree/master/training) | for loading/storing machine learning models and making model predictions for files put in the load_dir. | 
| [production](https://github.com/jim-schwoebel/Allie/tree/master/production) | a folder for outputting production-ready repositories via the YAML.py script. | 
| [tests](https://github.com/jim-schwoebel/Allie/tree/master/tests) | for running local tests and making sure everything works as expected. | 
| [train_dir](https://github.com/jim-schwoebel/Allie/tree/master/train_dir) | a directory where you can put in audio, text, image, video, or .CSV files in folders and train machine learning models from the model.py script in the ./training/ directory. |
| [training](https://github.com/jim-schwoebel/Allie/tree/master/models) | for training machine learning models via specified model training scripts. |

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
- **transcript definition flexibility** - transcripts can be audio, text, image, video, and csv transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). The schema above allows for a flexible definition for transcripts that can accomodate all forms. 
- **featurization flexibility** - many types of features can be put into this array of the same data type. For example, an audio file can be featurized with 'standard_features' and 'praat_features' without really affecting anything. This eliminates the need to re-featurize and reduces time to sort through multiple types of featurizations during the data cleaning process.
- **label annotation flexibility** - can take the form of ['classname_1', 'classname_2', 'classname_N...'] - classification problems and [{classname1: 'value'}, {classname2: 'value'}, ... {classnameN: 'valueN'}] where values are between [0,1] for regression problems. 
- **model predictions** - one survey schema can be used for making model predictions and updating the schema with these predictions. Note that any model that is used for training can be used to make predictions in the load_dir. 

We are currently in process to implement this schema into the SurveyLex architecture. 

## settings 

Settings can be modified in the settings.json file. If no settings.json file is identified, it will automatically be created with some default settings from the setup.py script.

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization | 'standard_features' | 'audioset_features', 'audiotext_features', 'librosa_features', 'meta_features', 'mixed_features', 'myprosody_features', 'praat_features', 'pspeech_features', 'pyaudio_features', 'sa_features', 'sox_features', 'specimage_features', 'specimage2_features', 'spectrogram_features', 'standard_features' | 
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization | 'nltk_features' | 'fast_features', glove_features, nltk_features, spacy_features, w2vec_features | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization | 'image_features' | 'image_features', 'inception_features', 'resnet_features', 'tesseract_features', 'vgg16_features', 'vgg19_features', 'xception_features' | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization | 'video_features' | 'video_features', 'y8m_features' | 
| [default_csv_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of csv features used for featurization | 'csv_features' | 'csv_features' | 
| bias_discovery | looks for biases in datasets during featurization (e.g. ages and genders) | False | True, False | 
| transcribe_audio | determines whether or not to transcribe an audio file via default_audio_transcriber | True | True, False | 
| default_audio_transcriber | the default audio transcriber if transcribe_audio == True | 'pocketsphinx' | 'pocketsphinx' | 
| transcribe_text | determines whether or not to transcribe a text file via default_text_transcriber | True | True, False | 
| default_text_transcriber | the default text transcriber if transcribe_text == True | 'raw text' | 'raw text' | 
| transcribe_image | determines whether or not to transcribe an image file via default_image_transcriber | True | True, False | 
| default_image_transcriber | the default image transcriber if transcribe_image == True | 'tesseract' | 'tesseract' | 
| transcribe_video | determines whether or not to transcribe a video file via default_video_transcriber | True | True, False | 
| default_video_transcriber | the default video transcriber if transcribe_video == True | 'tesseract_connected_over_frames' | 'tesseract_connected_over_frames' | 
| transcribe_csv | determines whether or not to transcribe a csv file via default_csv_transcriber | True | True, False | 
| default_csv_transcriber | the default video transcriber if transcribe_csv == True | 'raw text' | 'raw text' | 
| default_training_script | the specified traning script to train machine learning models | 'tpot' | 'scsr', 'tpot', 'hyperopt', 'keras', 'devol' or 'ludwig' | 
| augment data | specifies whether or not you'd like to augment data during training |  False | True, False | 
| visualize data | specifies whether or not you'd like to see a visualization during model training |  False | True, False | 
| create_YAML | specifies whether or not you'd like to output a production-ready repository for model deployment |  False | True, False | 
| model_compress | if True compresses the model for production purposes to reduce memory consumption. Note this only can happen on Keras or scikit-learn / TPOT models for now.| True | True, False | 

## License
This repository is licensed under a trade secret. Please do not share this code outside the core team.

## Feedback
Any feedback on the book or this repository is greatly appreciated. 
* If you find something that is missing or doesn't work, please consider opening a [GitHub issue](https://github.com/jim-schwoebel/Allie/issues).
* If you want to talk to me directly, please send me an email @ js@neurolex.co. 

## Additional resources
Here are some additional resources in this repository (not covered in the information above): 
* [Data augmentation](https://github.com/jim-schwoebel/Allie/tree/master/datasets/augmentation) - augmentation repositories for audio, text, and video files. 
* [Data cleaning](https://github.com/jim-schwoebel/Allie/tree/master/datasets/cleaning) - data cleaning scripts for deleting duplicate files and removing silence (for audio files).
