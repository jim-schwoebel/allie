# voice_modeling
Master repository for modeling voice files. Transformed from NLX-model.

1. Get dataset (assess bias). &#x2611;
2. Featurize and clean dataset (if True, clean according to data type). &#x2611;
3. feature selection (https://machinelearningmastery.com/feature-selection-machine-learning-python/). &#x2611;
4. modeling - SC, TPOT, Keras, Ludwig. &#x2611;
5. visualize models (Yellowbrick) - feature selection / etc. [only Ludwig] &#x2611;
6. compress models for production. &#x2611;
7. make predictions from all models (either compressed or not compressed). &#x2611;
8. automated testing. &#x2611;
9. server deployment (model compression, etc.). 
10. improved documentation w/ videos and whatnot. 

^^ make a quick visual above with a gif to show what this repo can do. 

^^ show some command line examples of what you can do with training [text, image, image, video, csv].

## getting started 

First, clone the repository:
```
git clone git@github.com:jim-schwoebel/voice_modeling.git
cd voice_modeling  
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

## folder structure

```
.. --> datasets 
.. --> features 
.. --> load_dir
.. --> models
.. --> production
.. --> tests
.. --> train_dir 
.. --> training 
```

## types of data

Load folders and data script type based on principal type of file.

* Audio --> .WAV / .MP3 (can transcribe) --> .WAV 
* Text --> .TXT / .PPT / .DOCX --> .TXT
* Images --> .PNG / .JPG (can transcribe images) --> .PNG 
* Video --> .MP4 / .M4A (can transcribe video, audio, and images) --> .MP4 
* CSV --> .CSV --> loads categorical and numerical data into .JSON 

This aligns well with how we define 'samples' in our pipeline.

## settings 

Settings can be modified in the settings.json file. If no settings.json file is identified, it will automatically be created with some default settings from the setup.py script. 

default_audio_features --> can be 'all' or any specific featurizer ('standard_features')

| setting | description | default setting | all options | 
|------|------|------|------| 
| default_audio_features | default set of audio features used for featurization | standard_features | audioset_features, audiotext_features, librosa_features, meta_features, mixed_features, myprosody_features, praat_features, pspeech_features, pyaudio_features, sa_features, sox_features, specimage_features, specimage2_features, spectrogram_features, standard_features | 
```
default_audio_features=audio_features
default_text_features=text_features
default_image_features=image_features
default_video_features=video_features
bias_discovery=True 
transcribe_audio=True 
default_audio_transcriber=pocketsphinx
transcribe_images=True 
transcribe_videos=True
default_training_script=keras
clean_data=True 
augment_data=False 
visualize_data=True 
create_YAML=True 
model_compress=True
```

Bias discovery looks at all the audio files and plots out a bias assessment before modeling. This can help identify areas of the dataset that may need to be augmented before modeling and can work across any type. 
* solution = class pairing (equal delete)
* solution = data augmentation (to make one class more represented) / combining with other datasets 

Transcription can happen for audio, image, or video datasets. For audio a standard speech-to-text model (e.g. pocketsphinx) can be used. For image and video, it is assumed an OCR transcription can happen (videos are sampled at some frequency then transcripts are stitched together). 

Default_training script = the type of script used for training. Can be simple, tpot, autokeras, or ludwig.  

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

Create YAML means that the entire repository will be generated to host the model for production. 

Model compression if True compresses the model for production purposes to reduce memory consumption. Note this only can happen on Keras or scikit-learn / TPOT models for now.

## References 
### Bias discovery
* apply machine learning models for each types of data (audio, text, images, video, .CSV) to auto detect things like ages, genders, etc. 

### Datasets
* [PyDataset](https://github.com/iamaziz/PyDataset) - numerical datasets
* [CommonVoice]() - download link to Common Voice - standard dataset 
* [AudioSet_download]() - download link to AudioSet 

### Visualization
* [Yellowbrick](https://www.scikit-yb.org/en/latest/) - for features. 
* [Bokeh]() - for features.
* [Matplotlib]() - for features. 

### Modeling 
* TPOT
* AutoKeras
* Luwdig  

### Model compression
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor) - should be good for Keras.
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration) - good papers
* [PocketFlow]() - everything else (Tencent's data science team).
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble) - should be good for TPOT I believe.
