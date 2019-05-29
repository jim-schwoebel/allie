# voice_modeling
Master repository for modeling voice files. Transformed from NLX-model.

1. Get dataset
2. feature selection (https://machinelearningmastery.com/feature-selection-machine-learning-python/) 
3. modeling (if < 1000 then simple classification, if >1000 deep learning techniques - iterate through various architectures).
4. apply models. 

# types of data
* voice data
** features in voicebook
* text data
** nltk featurize
** spacy featurize 
** word2vec featurize 
* image data 
** edge detection 
* video data
* other data (numerical / categorical variables)

(auto detect input from file type, then apply various algorithms). 

## problems looked at 
* accent detection
* race detection 
* gender detection
* age detection
* stress detection
* emotion detection (face images) 

## list of all applicable models now
* accuracies, standard deviations (TPOT) 

## labeling
* [sound_event_detection]

## featurization scripts (parallel)
### Audio
* [DisVoice](https://github.com/jcvasquezc/DisVoice)
* [AudioOwl](https://github.com/dodiku/AudioOwl)

### Text
* []()

### Image
* []()

### Video 
* []()

## modeling 
* [TPOT]()
* [Ludwig]()

## visualization
* [Yellowbrick]()
