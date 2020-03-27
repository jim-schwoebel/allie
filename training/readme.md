## Training scripts 

Quickly train according to the default_training_script using model.py.

![](https://github.com/jim-schwoebel/Allie/blob/master/training/helpers/train.gif)

## Getting started

All you need to do to get started is go to this repository and run model.py:

```
cd allie/training
python3 model.py 
```

You then will be asked a few questions regarding the training process (in terms of data type, number of classes, and the name of the model). Note that --> indicates typed responses. 

```
what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)
--> 1

 OK cool, we got you modeling audio files 

how many classes would you like to model? (2 available) 
--> 2
these are the available classes: 
['one', 'two']
what is class #1 
--> one
what is class #2 
--> two
what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) 
--> test
is this a classification (c) or regression (r) problem? 
--> c
```

After this, the model will be trained and placed in the models/[sampletype_models] directory. For example, if you trained an audio model with TPOT, the model will be placed in the allie/models/audio_models/ directory. 

For automated training, you can alternatively pass through sys.argv[] inputs as follows:

```
python3 model.py audio 2 c male female
```
Where:
- audio = audio file type 
- 2 = 2 classes 
- c = classification (r for regression)
- male = first class
- female = second class [via N number of classes]

Now you're ready to go to load these models and [make predictions](https://github.com/jim-schwoebel/allie/tree/master/models).

## Default_training scripts 

There are 6 potential training script settings (customized in the 'settings.json'). Recommended setting is TPOT.

* '[devol](https://github.com/joeddav/devol)' - genetic programming keras cnn layers. &#x2611;
* '[hypsklearn](https://github.com/hyperopt/hyperopt-sklearn)' - seems stable - hyperparameter optimization of the data. &#x2611;
* '[hungabunga](https://github.com/ypeleg/HungaBunga)' - brute-Force all sklearn models with all parameters using .fit .predict &#x2611;
* '[keras](https://keras.io/getting-started/faq/)' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* '[ludwig](https://github.com/uber/ludwig)' - deep learning (simple ludwig). - convert every feature to numerical data. &#x2611; 
* '[neuraxle](https://github.com/Neuraxio/Neuraxle)' - A Sklearn-like Framework for Hyperparameter Tuning and AutoML in Deep Learning projects. &#x2611;
* '[scsr](https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py)' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* **'[tpot](https://epistasislab.github.io/tpot/)'** - TPOT classification / regression (autoML). &#x2611;

Note some of the deep learning autoML techniques can take days for optimization, and there are compromises in accuracy vs. speed in training.

## Actively working on (in future)

### model training 
* [MLBlocks](https://github.com/HDI-Project/MLBlocks) - most recent framework @ MIT
* [SAFE](https://github.com/ModelOriented/SAFE) - black box trainer / can improve scsr with this
* [BTB](https://github.com/HDI-Project/BTB) - hyperparameter tuning framework 
* 'keras-inference-time-optimizer](https://github.com/ZFTurbo/Keras-inference-time-optimizer) - restructure keras neural network to reduce inference time without reducing accuracy.
* '[keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)' - 510x reduced memory footprint training keras models.
* '[adanet](https://github.com/tensorflow/adanet)' - Google's AutoML framework in tensorflow (https://github.com/tensorflow/adanet).
* '[alphapy](https://alphapy.readthedocs.io/en/latest/user_guide/pipelines.html#model-object-creation)' - keras, scikit-learn, xgboost - highly customizable setttings for data science pipelines and feature selection. 
* '[BTB](https://github.com/HDI-Project/BTB)' - BTB for hyperparameter tuning
* '[Hyperband-scikitlearn](https://github.com/thuijskens/scikit-hyperband)' - implements a class HyperbandSearchCV that works exactly as GridSearchCV and RandomizedSearchCV from scikit-learn do, except that it runs the hyperband algorithm under the hood.
* '[gentun](https://github.com/gmontamat/gentun)' - genetic algorithm approach with distributed training capability.
* '[python-sherpa](https://github.com/sherpa-ai/sherpa)' - sherpa bayesian hyperparameter optimization
* '[imbalance-learn](https://pypi.org/project/imbalanced-learn/)' - imbalance learn different ML techniques.
* Add in RFE (https://www.scikit-yb.org/en/latest/_modules/yellowbrick/features/rfecv.html) and hyperparameter optimization to scsr with hyperopt-sklearn (https://github.com/hyperopt/hyperopt-sklearn). 
* Ludwig (features) - need to be able to featurize according to image - path to image and text types - transcripts
* Add in video and image augmentation scripts via Google and others.

### Other stuff 
* [PocketFlow](https://github.com/Tencent/PocketFlow) - allow for ludwig model compression.

### Data modeling tutorials (Readmes)
* Audio file training example
* Text file training example 
* Image file training example
* Video file training example 
* CSV file training example

## Additional documentation
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor)
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [Model compression](https://www.slideshare.net/AnassBensrhirDatasci/deploying-machine-learning-models-to-production)
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble)
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)
