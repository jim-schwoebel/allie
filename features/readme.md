## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## Standard feature dictionary (.JSON)

Show outline for this below. This makes it flexible to any featurization and transcript type.

```
def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		    'text': dict(),
		    'image':dict(),
		    'video':dict(),
		    'csv': dict(),
		    }
            
 	transcripts={'audio':dict(),
		      'text': dict(),
		      'image': dict(),
		      'video':dict(),
		      }

	data={'sampletype': sampletype,
		'features': features,
		'transcripts': transcripts,
		'labels': []}
        
```

Note that there can be audio transcripts, image transcripts, and video transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). The schema above allows for a flexible definition for transcripts that can accomodate all forms. 

Sampletype = 'audio', 'text', 'image', 'video', 'csv'

Note that only .CSV files may have audio, text, image, video features all-together (as the .CSV can contain files in a current directory that need to be featurized together). Otherwise, audio files likely will have audio features, text files will have text features, image files will have image features, and video files will have video features. 

## Implemented 

Note that all scripts implemented have features and their corresponding labels. It is important to provide labels to understand what the features correspond to. It's also to keep in mind the relative speeds of featurization to optimize server costs (they are provided here for reference).

### Audio
* [audioset_features]()
* [librosa_features]()
* [meta_features]()
* [myprosody_features]() - sometimes unstable 
* [praat_features]()
* [pspeech_features]() 
* [pyaudio_features]()
* [sa_features]()
* [sox_features]()
* [standard_features]() - standard audio feature array (default)
* [spectrogram_features]() 
* [specimage_features]()
* [specimage2_features]()

### Text
* [fast_features]()
* [glove_features]() 
* [nltk_features]() - standard text feature array (default)
* [spacy_features]() 
* [w2vec_features]() 

### Images 
* [image_features]() - standard image feature array (default)
* [inception_features]() 	
* [resnet_features]()
* [tesseract_features]()	
* [vgg16_features]() 
* [vgg19_features]() 
* [xception_features]() 

### Videos 
* [video_features]() - standard video feature array (default)
* [y8m_features]() 

### CSV 
* [csv_features]() - standard CSV feature array

## Not Implemented / Work in progress
### Audio
* Add in transcription to standard audio array if settings.JSON audio transcript == True; customize transcription types.
* [text_classify]() - transcript_features() - text features (nltk) 
* [mixed_features]() - mixed_features() - ratios 
* [audiotext_classify]() - audiotext_classify() - audio and text embeddings together 
* [kaldi features](https://github.com/pykaldi/pykaldi)  - GMM and other such features. https://pykaldi.github.io/api/kaldi.feat.html#module-kaldi.feat.fbank
* [Noise separation](https://github.com/seanwood/gcc-nmf) - noise separation technique.
* [Make noisy](https://github.com/Sato-Kunihiko/audio-SNR/) - noisy add-on.
* [Speaker diarization](https://github.com/faroit/CountNet) - counting # of speakers and shifts (CountNet). 
* [pLDA](https://github.com/RaviSoji/plda) - implement pLDA for speech mfcc coefficients in window lengths for speaker recognition / i Vectors. - noise (https://www.isca-speech.org/archive/interspeech_2015/papers/i15_2317.pdf) - implement with librosa in 20 ms timescale (frames).

### Text
* add in text transcriptoin (default value) 
* input text files 
* BERT pre-trained model - https://github.com/huggingface/pytorch-pretrained-BERT

### Images 
* Add in transcription to standard image array if settings.JSON image transcript == True.

### Videos 
* add in transcription to the standard video array {'transcript': video_transcript, 'type': video} if settings.JSON video transcript == True.
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* [Near duplicate](https://github.com/Chinmay26/Near-Duplicate-Video-Detection)

### CSV 
* be able to determine file type and featurize accordingly on local path ./img.jpg ,./audio.wav, ./video.mp4, ./text.txt, etc.; these will then be featurized with default featurizers for images, audio, video, and text accordingly.
* accomodate all types on ludwig (https://uber.github.io/ludwig/getting_started/) - binary, numerical, category, set, bag, sequence, text, timeseries, image
* [seglearn](https://github.com/dmbee/seglearn) - time series pipeline.

