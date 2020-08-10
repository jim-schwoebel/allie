## Featurization scripts

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/featurize.png)

This is a folder for extracting features from audio, text, image, video, or .CSV files. This is done via a convention for featurizers, which are in the proper folders (e.g. audio files --> audio_features). In this way, we can appropriately create featurizers for various sample data types according to the default featurizers as specified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json).

Note that as part of the featurization process, files can also be transcribed by default transcription types as specified in [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json).

## How to featurize folders of files 

To featurize an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```
cd ~ 
cd allie/features/audio_features
python3 featurize.py /Users/jimschwoebel/allie/load_dir
```

The code above will featurize all the audio files in the folderpath via the default_featurizer specified in the settings.json file (e.g. 'standard_features'). 

If you'd like to use a different featurizer you can specify it optionally:

```
cd ~ 
cd allie/features/audio_features
python3 featurize.py /Users/jimschwoebel/allie/load_dir librosa_features
```

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/audio_features | 
| text files | .TXT | ```python3 featurize.py [folderpath]``` | ./allie/features/text_features| 
| image files | .PNG | ```python3 featurize.py [folderpath]``` | ./allie/features/image_features | 
| video files | .MP4 | ```python3 featurize.py [folderpath]``` |./allie/features/video_features | 
| csv files | .CSV | ```python3 featurize.py [folderpath]``` | ./allie/features/csv_features | 

## [Standard feature dictionary (.JSON)](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py)

This is the standard feature array to accomodate all types of samples (audio, text, image, video, or CSV samples):

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
	
	# getting settings can be useful to see if settings are the same in every
	# featurization, as some featurizations can rely on certain settings to be consistent
	prevdir=prev_dir(os.getcwd())
	try:
		settings=json.load(open(prevdir+'/settings.json'))
	except:
		# this is for folders that may be 2 layers deep in train_dir
		settings=json.load(open(prev_dir(prevdir)+'/settings.json'))
	
	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
		  'models': models,
		  'labels': [],
		  'errors': [],
		  'settings': settings,
		 }
	
	return data
```
There are many advantages for having this schema including:
- **sampletype definition flexibility** - flexible to 'audio' (.WAV / .MP3), 'text' (.TXT / .PPT / .DOCX), 'image' (.PNG / .JPG), 'video' (.MP4), and 'csv' (.CSV). This format can also can adapt into the future to new sample types, which can also tie to new featurization scripts. By defining a sample type, it can help guide how data flows through model training and prediction scripts. 
- **transcript definition flexibility** - transcripts can be audio, text, image, video, and csv transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). You can also add multiple transcripts (e.g. Google and PocketSphinx) for the same sample type.
- **featurization flexibility** - many types of features can be put into this array of the same data type. For example, an audio file can be featurized with 'standard_features' and 'praat_features' without really affecting anything. This eliminates the need to re-featurize and reduces time to sort through multiple types of featurizations during the data cleaning process.
- **label annotation flexibility** - can take the form of ['classname_1', 'classname_2', 'classname_N...'] - classification problems and [{classname1: 'value'}, {classname2: 'value'}, ... {classnameN: 'valueN'}] where values are between [0,1] for regression problems. 
- **model predictions** - one survey schema can be used for making model predictions and updating the schema with these predictions. Note that any model that is used for training can be used to make predictions in the load_dir. 
- **visualization flexibility** - can easily visualize features of any sample type through Allie's [visualization script](https://github.com/jim-schwoebel/allie/tree/master/visualize) (e.g. tSNE plots, correlation matrices, and more).
- **error tracing** - easily trace errors associated with featurization and/or modeling to review what is happening during a session.

This schema is inspired by [D3M-schema](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md) by the MIT media lab.

## Example

To featurize folders of files, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd features/audio_features
python3 featurize.py /Users/jim/desktop/allie/train_dir/males/
```

What results are featurized audio files in .JSON format, following the [standard dictionary](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py). In this case, librosa_features ahve been added to the audio features dictionary element:
```
{"sampletype": "audio", "transcripts": {"audio": {"deepspeech_dict": "no time to fear with which segnatura"}, "text": {}, "image": {}, "video": {}, "csv": {}}, "features": {"audio": {"librosa_features": {"features": [26.0, 91.6923076923077, 51.970234995079515, 168.0, 3.0, 91.5, 151.99908088235293, 1.4653875809266168, 1.034836765896441, 6.041317273120036, 0.0, 1.1416813305224416, 1.0, 0.0, 1.0, 1.0, 1.0, 0.8034238194742891, 0.011214064753885858, 0.825403850923042, 0.7916342653642267, 0.799591131419195, 0.6655924418356907, 0.011330648308197012, 0.6913809809074571, 0.6541308272200052, 0.661418338298795, 0.6733986003222069, 0.015596609024721262, 0.7120299672460608, 0.6587366189344313, 0.6663189274679151, 0.6465513378063461, 0.025089929067247278, 0.7051369537517457, 0.6222968652297333, 0.6355192254778563, 0.6149336085508313, 0.018343891143310215, 0.6627315206635452, 0.5977577122027029, 0.607582407218563, 0.647661411671485, 0.008459482998415297, 0.6713073144386539, 0.6386408867240191, 0.6452831890619676, 0.699964101062811, 0.007615472147513634, 0.7189362743395604, 0.6881770842137894, 0.6979190256767938, 0.7291497355453966, 0.00940911939072975, 0.7401396411729065, 0.7017699200214637, 0.7331228695983505, 0.7531264693954148, 0.01329521814226255, 0.763276239791762, 0.7116522984632485, 0.7590587373746718, 0.7350729652806612, 0.03153112064046178, 0.7660970571880834, 0.657206917912684, 0.7470611927003584, 0.6322639720282205, 0.017938497171685913, 0.6615311610839943, 0.5941319788201291, 0.6343007353689181, 0.5778206400146562, 0.017037307606750474, 0.6125509247436416, 0.5514687732413751, 0.5754841853855134, -201.3533386630615, 47.15330192057996, -102.18535523700777, -502.49179220012513, -201.31156601864114, 125.25603463854729, 18.082299070656465, 162.0310670688952, 63.261254376704564, 127.81548416353635, -71.79877076064429, 25.856723357072436, 0.00849312238186889, -135.65333506936406, -70.68911674995219, 36.022947214229916, 17.88923654191366, 88.64728797287137, -4.5486056006807685, 32.97698049436967, -43.81875187482729, 14.559312843201848, -11.695474808994188, -92.5127540007665, -43.3715458030925, -2.8686813657134924, 15.41813093478462, 45.52354221029802, -46.72785054356835, -3.558833340504383, -27.922593394058616, 10.927878425731766, -1.4375539667895811, -55.265896216864945, -26.81749586891489, -5.505759781242237, 11.987368006311469, 29.15642000598597, -32.415122640721314, -5.511091433999935, -15.823860504423996, 9.451418989514549, 5.241598458345977, -37.9588985258688, -15.755382749864523, -0.4895271220805938, 9.193396110591832, 21.927443663882503, -17.64191390126784, -1.8424862936634168, -9.022651931057775, 9.128291169735268, 18.549517520727974, -30.5660411806481, -8.34285803735538, -9.278437145276117, 8.646155204335475, 14.76706895840456, -27.949477962308606, -9.825154645141577, -0.5851781562774249, 7.5172603841211965, 15.958013214521532, -19.188487080387258, -1.4245433010743693, -0.0002136159172391847, 0.0001244699595478107, -4.3705042965701695e-06, -0.0006147477962819087, -0.00018150223332853152, 1.798108486545567, 1.0501312811041865, 5.152526911465508, 0.0364132947716482, 1.5099976502325472, 1974.162593266179, 480.2110921639575, 3767.7561717054723, 1207.1282130698876, 1914.9902608895832, 1597.477406458397, 300.1835303257206, 2611.5607852021894, 1096.7664088833956, 1524.2274402971789, 19.140024880982267, 5.679892757483778, 63.887253040125415, 7.787955276296776, 18.524605940498716, 0.00018343908595852554, 0.0003219500940758735, 0.003092152765020728, 9.228861017618328e-06, 7.437301246682182e-05, 3716.331787109375, 970.2260872230195, 6556.8603515625, 1894.921875, 3692.9443359375, 0.11654459635416667, 0.0477755604304587, 0.28662109375, 0.04052734375, 0.1044921875, 0.07000089436769485, 0.042519740760326385, 0.18653172254562378, 0.010408219881355762, 0.06168703734874725], "labels": ["onset_length", "onset_detect_mean", "onset_detect_std", "onset_detect_maxv", "onset_detect_minv", "onset_detect_median", "tempo", "onset_strength_mean", "onset_strength_std", "onset_strength_maxv", "onset_strength_minv", "onset_strength_median", "rhythm_0_mean", "rhythm_0_std", "rhythm_0_maxv", "rhythm_0_minv", "rhythm_0_median", "rhythm_1_mean", "rhythm_1_std", "rhythm_1_maxv", "rhythm_1_minv", "rhythm_1_median", "rhythm_2_mean", "rhythm_2_std", "rhythm_2_maxv", "rhythm_2_minv", "rhythm_2_median", "rhythm_3_mean", "rhythm_3_std", "rhythm_3_maxv", "rhythm_3_minv", "rhythm_3_median", "rhythm_4_mean", "rhythm_4_std", "rhythm_4_maxv", "rhythm_4_minv", "rhythm_4_median", "rhythm_5_mean", "rhythm_5_std", "rhythm_5_maxv", "rhythm_5_minv", "rhythm_5_median", "rhythm_6_mean", "rhythm_6_std", "rhythm_6_maxv", "rhythm_6_minv", "rhythm_6_median", "rhythm_7_mean", "rhythm_7_std", "rhythm_7_maxv", "rhythm_7_minv", "rhythm_7_median", "rhythm_8_mean", "rhythm_8_std", "rhythm_8_maxv", "rhythm_8_minv", "rhythm_8_median", "rhythm_9_mean", "rhythm_9_std", "rhythm_9_maxv", "rhythm_9_minv", "rhythm_9_median", "rhythm_10_mean", "rhythm_10_std", "rhythm_10_maxv", "rhythm_10_minv", "rhythm_10_median", "rhythm_11_mean", "rhythm_11_std", "rhythm_11_maxv", "rhythm_11_minv", "rhythm_11_median", "rhythm_12_mean", "rhythm_12_std", "rhythm_12_maxv", "rhythm_12_minv", "rhythm_12_median", "mfcc_0_mean", "mfcc_0_std", "mfcc_0_maxv", "mfcc_0_minv", "mfcc_0_median", "mfcc_1_mean", "mfcc_1_std", "mfcc_1_maxv", "mfcc_1_minv", "mfcc_1_median", "mfcc_2_mean", "mfcc_2_std", "mfcc_2_maxv", "mfcc_2_minv", "mfcc_2_median", "mfcc_3_mean", "mfcc_3_std", "mfcc_3_maxv", "mfcc_3_minv", "mfcc_3_median", "mfcc_4_mean", "mfcc_4_std", "mfcc_4_maxv", "mfcc_4_minv", "mfcc_4_median", "mfcc_5_mean", "mfcc_5_std", "mfcc_5_maxv", "mfcc_5_minv", "mfcc_5_median", "mfcc_6_mean", "mfcc_6_std", "mfcc_6_maxv", "mfcc_6_minv", "mfcc_6_median", "mfcc_7_mean", "mfcc_7_std", "mfcc_7_maxv", "mfcc_7_minv", "mfcc_7_median", "mfcc_8_mean", "mfcc_8_std", "mfcc_8_maxv", "mfcc_8_minv", "mfcc_8_median", "mfcc_9_mean", "mfcc_9_std", "mfcc_9_maxv", "mfcc_9_minv", "mfcc_9_median", "mfcc_10_mean", "mfcc_10_std", "mfcc_10_maxv", "mfcc_10_minv", "mfcc_10_median", "mfcc_11_mean", "mfcc_11_std", "mfcc_11_maxv", "mfcc_11_minv", "mfcc_11_median", "mfcc_12_mean", "mfcc_12_std", "mfcc_12_maxv", "mfcc_12_minv", "mfcc_12_median", "poly_0_mean", "poly_0_std", "poly_0_maxv", "poly_0_minv", "poly_0_median", "poly_1_mean", "poly_1_std", "poly_1_maxv", "poly_1_minv", "poly_1_median", "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_maxv", "spectral_centroid_minv", "spectral_centroid_median", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_bandwidth_maxv", "spectral_bandwidth_minv", "spectral_bandwidth_median", "spectral_contrast_mean", "spectral_contrast_std", "spectral_contrast_maxv", "spectral_contrast_minv", "spectral_contrast_median", "spectral_flatness_mean", "spectral_flatness_std", "spectral_flatness_maxv", "spectral_flatness_minv", "spectral_flatness_median", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_rolloff_maxv", "spectral_rolloff_minv", "spectral_rolloff_median", "zero_crossings_mean", "zero_crossings_std", "zero_crossings_maxv", "zero_crossings_minv", "zero_crossings_median", "RMSE_mean", "RMSE_std", "RMSE_maxv", "RMSE_minv", "RMSE_median"]}}, "text": {}, "image": {}, "video": {}, "csv": {}}, "models": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "labels": ["females"], "errors": [], "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": true, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/featurize.gif)](https://drive.google.com/file/d/1YQXgnAvKHVjSgYteIFz-9H0znr1UKCLl/view?usp=sharing)

## Implemented Features

Note that all scripts implemented have features and their corresponding labels. It is important to provide labels to understand what the features correspond to. It's also to keep in mind the relative speeds of featurization to optimize server costs (they are provided here for reference).

### [Audio](https://github.com/jim-schwoebel/allie/tree/master/features/audio_features)
* [audioset_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audioset_features.py) - 
simple script to extract features using the VGGish model released by Google.
* [audiotext_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audiotext_features.py) - featurizes data with text feautures extracted from the transcript.
* [librosa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/librosa_features.py) - 
extracts acoustic features using the [LibROSA library](https://librosa.org/).
* [loudness_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/loudness_features.py) - extracts loudness features using the [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) library.
* [meta_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/meta_features.py) - extracts meta features from models trained on the audioset dataset.
* [mixed_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/mixed_features.py) - random combinations of audio and text features (via ratios).
* [multispeaker_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/multispeaker_features.py) - detect number of speakers in audio files using the [CountNet](https://github.com/faroit/CountNet) deep learning model.
* [opensmile_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/opensmile_features.py) - 14 embeddings with [OpenSMILE](https://www.audeering.com/opensmile/) possible here; defaults to GeMAPSv01a.conf.
* [praat_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/praat_features.py) - extracts features from the [parselmouth.praat library](https://pypi.org/project/praat-parselmouth/).
* [prosody_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/prosody_features.py) - prosody using Google's VAD - including pause length, total number of pauses, and pause variability.
* [pspeech_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pspeech_features.py) - extracts features with the [python_speech features library](https://github.com/jameslyons/python_speech_features).
* [pyaudio_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyaudio_features.py) - extract features withh the [pyaudioanalysis](https://github.com/tyiannak/pyAudioAnalysis) library.
* [pyaudiolex_features](https://github.com/tyiannak/pyAudioAnalysis) - time series features extracted with the [pyaudioanalysis](https://github.com/tyiannak/pyAudioAnalysis) library.
* [pyworld_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyworld_features.py) - f0 and and spectrogram features.
* [sa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sa_features.py) - some additional features extracted using the [SignalAnalysis](https://brookemosby.github.io/Signal_Analysis/Signal_Analysis.features.html#module-Signal_Analysis.features.signal) library.
* [sox_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sox_features.py) - features extracted from the [sox](http://sox.sourceforge.net/sox.html) command line interface.
* [speechmetrics_features](https://github.com/aliutkus/speechmetrics) - extracts features that estimate speech quality without references using the [speechmetrics](https://github.com/aliutkus/speechmetrics) library.
* [specimage_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage_features.py) - image-based features from spectrograms.
* [specimage2_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage2_features.py) - image-based features from spectrograms (alternative).
* [spectrogram_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/spectrogram_features.py) - spectrogram-based features.
* [standard_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/standard_features.py) - standard audio feature array (default).

### [Text](https://github.com/jim-schwoebel/allie/tree/master/features/text_features)
* [bert features](https://github.com/UKPLab/sentence-transformers) - extract BERT-related features from sentences (note shorter sentences run faster here, and long text can lead to long featurization times).
* [fast_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/fast_features.py) - features extracted using the [FastText model](https://fasttext.cc/).
* [glove_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/glove_features.py) - features extracted using the [GloVe model](https://nlp.stanford.edu/projects/glove/).
* [grammar_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/grammar_features.py) - 85k+ grammar features (memory intensive)
* [nltk_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/nltk_features.py) - standard text feature array (default).
* [spacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/spacy_features.py) - feature extracted with the [SpaCy library](https://spacy.io/).
* [textacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/textacy_features.py) - a variety of document classification and topic modeling features.
* [text_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/text_features.py) - many different types of features like emotional word counts, total word counts, Honore's statistic and others.
* [w2v_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/w2vec_features.py) - note this is the largest model from Google and may crash your computer if you don't have enough memory. I'd recommend fast_features if you're looking for a pre-trained embedding.

### [Image](https://github.com/jim-schwoebel/allie/tree/master/features/image_features)
* [image_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/image_features.py) - standard image feature array (default).
* [inception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/inception_features.py) - features extracted with the [Inception model](https://keras.io/api/applications/inceptionv3/).
* [resnet_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/resnet_features.py) - features extracted with the [ResNet model](https://keras.io/api/applications/resnet/#resnet50v2-function).
* [squeezenet_features](https://github.com/rcmalli/keras-squeezenet) - features extracted with the [Squeezenet model](https://github.com/forresti/SqueezeNet); this has an efficient memory footprint.
* [tesseract_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/tesseract_features.py) - features extracted with OCR on images using the [pytesseract module](https://pypi.org/project/pytesseract/).
* [vgg16_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg16_features.py) - features extracted with hte [VGG16 model](https://keras.io/api/applications/vgg/#vgg16-function).
* [vgg19_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg19_features.py) - features extracted with hte [VGG19 model](https://keras.io/api/applications/vgg/#vgg19-function).
* [xception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/xception_features.py) - features extracted with hte [Xception model](https://keras.io/api/applications/xception/).

### [Video](https://github.com/jim-schwoebel/allie/tree/master/features/vide_features)
* [video_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/video_features.py) - standard video feature array (default) - extracts acoustic, linguistic, and video features.
* [y8m_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/y8m_features.py) - extracts acoustic, linguistic, and video features using the Y8M model.

### CSV 

.CSV can include numerical data, categorical data, audio files (./audio.wav), image files (.png), video files (./video.mp4), text files ('.txt' or text column), or other .CSV files. This scope of a table feature is inspired by [D3M schema design proposed by the MIT data lab](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md).

* [featurize_csv_regression](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/featurize_csv_regression.py) - standard CSV feature array that can accomodate audio, image, video, text, and numerical data formats.

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

Here are some features settings that can be customized with Allie's API. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| version | version of Allie release | 1.0 | 1.0 |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| default_text_transcriber | the default transcription techniques used to parse raw .TXT files during model training| ["raw_text"] | ["raw_text"]  | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| default_video_transcriber | the default transcription technique used for videos (.mp4 --> text from the video) | ["tesseract (averaged over frames)"] | ["tesseract (averaged over frames)"] |
| test_size | a setting that specifies the size of the testing dataset for defining model performance after model training. | 0.10 | Any number 0.10-0.50 | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
| transcribe_csv | a setting to define whether or not to transcribe csv files during featurization and model training via the default_csv_transcriber | True | True, False | 
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_text | a setting to define whether or not to transcribe text files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_video | a setting to define whether or not to transcribe video files during featurization and model training via the default_video_transcriber | True | True, False | 
