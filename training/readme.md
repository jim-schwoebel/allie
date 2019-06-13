## Settings

Training scripts here. 7 potential settings. Recommended setting is TPOT.
* **'[tpot](https://epistasislab.github.io/tpot/)'** - TPOT classification / regression (autoML). &#x2611;
* '[scsr](https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py)' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* '[keras](https://keras.io/getting-started/faq/)' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* '[devol](https://github.com/joeddav/devol)' - genetic programming keras cnn layers. &#x2611;
* '[ludwig](https://github.com/uber/ludwig)' - deep learning (simple ludwig). - convert every feature to numerical data. &#x2611; 
* '[adanet](https://github.com/tensorflow/adanet)' - Google's AutoML framework in tensorflow (https://github.com/tensorflow/adanet).
* '[alphapy](https://alphapy.readthedocs.io/en/latest/user_guide/pipelines.html#model-object-creation)' - keras, scikit-learn, xgboost (https://github.com/ScottfreeLLC/AlphaPy).

Note some of the deep learning autoML techniques can take days for optimization, and there are compromises in accuracy vs. speed in training.

## Things to finish (before production build)
Other things. 
* If augment == True, can augment (audio, text, image, and video data)
* Ludwig (features) - need to be able to featurize according to image - path to image and text types - transcripts
* [recursive feature elimination]() - can help select appropriate features / show feature importances (sc_ script) - use Yellowbrick for this.
* add in [featuretools](https://github.com/Featuretools/featuretools) to create higher-order features to get better accuracy.
* hyperparameter optimization - https://github.com/autonomio/talos
* Compress models? --> if True, compress with AutoKeras or scikit-learn compress (faster predictions from features) - https://github.com/Tencent/PocketFlow
* Create YAML file (for GitHub repository) - if True, select a cool .GIF to put in repo readme (docker container, featurizers, automated testing w/ test file) 
* Clustering algorithms 

## Data modeling tutorials (Readmes)
* Audio file training example
* Text file training example 
* Image file training example
* Video file training example 
* CSV file training example

## Future additions
* [Age/gender](https://towardsdatascience.com/predict-age-and-gender-using-convolutional-neural-network-and-opencv-fd90390e3ce6) - age and gender detection from images 

## Additional documentation
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)
