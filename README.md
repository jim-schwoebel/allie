# voice_modeling
Master repository for modeling voice files. Transformed from NLX-model.

1. Get dataset (assess bias).
2. feature selection (https://machinelearningmastery.com/feature-selection-machine-learning-python/). 
3. modeling - SC, TPOT, Keras, Ludwig.
4. visualize models (Yellowbrick) - feature selection / etc. 
5. compress models for production.
6. server deployment. 

## getting started 
```
git clone
cd voice_modeling 
python3 setup.py 
```

## settings 

settings.json

default_audio_features --> can be 'all' or any specific featurizer ('standard_features')

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
augment_data=True 
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

## types of data

Load folders and data script type based on principal type of file.

* Audio --> .WAV / .MP3 (can transcribe) --> .WAV 
* Text --> .TXT / .PPT / .DOCX --> .TXT
* Images --> .PNG / .JPG (can transcribe images) --> .PNG 
* Video --> .MP4 / .M4A (can transcribe video, audio, and images) --> .MP4 
* CSV --> .CSV --> loads categorical and numerical data into .JSON 

This aligns well with how we define 'samples' in our pipeline.

## Features to add
### Bias discovery
* apply machine learning models for each types of data (audio, text, images, video, .CSV) to auto detect things like ages, genders, etc. 

### Datasets
* [PyDataset](https://github.com/iamaziz/PyDataset) - numerical datasets
* [CommonVoice]() - download link to Common Voice - standard dataset 
* [AudioSet_download]() - download link to AudioSet 

### Visualization
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)

### Modeling 
* TPOT
* AutoKeras
* Luwdig  

### Model compression
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor) - should be good for Keras.
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble) - should be good for TPOT I believe.
