## Training scripts 

Quickly train according to the default_training_script using model.py.

![](https://github.com/jim-schwoebel/Allie/blob/master/training/helpers/train.gif)

## Default_training scrips 

There are 6 potential training script settings (customized in the 'settings.json'). Recommended setting is TPOT.
* **'[tpot](https://epistasislab.github.io/tpot/)'** - TPOT classification / regression (autoML). &#x2611;
* '[scsr](https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py)' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* '[keras](https://keras.io/getting-started/faq/)' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* '[devol](https://github.com/joeddav/devol)' - genetic programming keras cnn layers. &#x2611;
* '[ludwig](https://github.com/uber/ludwig)' - deep learning (simple ludwig). - convert every feature to numerical data. &#x2611; 
* '[hypsklearn](https://github.com/hyperopt/hyperopt-sklearn)' - seems stable - hyperparameter optimization of the data. &#x2611;

Note some of the deep learning autoML techniques can take days for optimization, and there are compromises in accuracy vs. speed in training.

## Actively working on (in future)

### model training 
* '[adanet](https://github.com/tensorflow/adanet)' - Google's AutoML framework in tensorflow (https://github.com/tensorflow/adanet).
* '[alphapy](https://alphapy.readthedocs.io/en/latest/user_guide/pipelines.html#model-object-creation)' - keras, scikit-learn, xgboost - highly customizable setttings for data science pipelines and feature selection. 
* Add in RFE (https://www.scikit-yb.org/en/latest/_modules/yellowbrick/features/rfecv.html) and hyperparameter optimization to scsr with hyperopt-sklearn (https://github.com/hyperopt/hyperopt-sklearn). 
* Ludwig (features) - need to be able to featurize according to image - path to image and text types - transcripts
* If clean == True, then clean each folder. This will remove any duplicate files (through byte-wise analysis). 
* If augment == True, can augment via augmentation scripts (audio, text, image, and video data).

### Other stuff 
* [PocketFlow](https://github.com/Tencent/PocketFlow) - allow for ludwig model compression.

## Data modeling tutorials (Readmes)
* Audio file training example
* Text file training example 
* Image file training example
* Video file training example 
* CSV file training example

## Additional documentation
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [Model compression](https://www.slideshare.net/AnassBensrhirDatasci/deploying-machine-learning-models-to-production) - overview of what is important for ML models in production.
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor) - should be good for Keras.
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration) - good papers
* [PocketFlow]() - everything else (Tencent's data science team).
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble) - should be good for TPOT I believe.
