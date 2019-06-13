## Settings

Training scripts here. 7 potential settings. Recommended setting is TPOT.
* **'tpot'** - TPOT classification / regression (autoML). &#x2611;
* 'scsr' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* 'keras' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* 'devol' - genetic programming keras (https://github.com/joeddav/devol.git). &#x2611;
* 'ludwig' - deep learning (simple ludwig). - convert every feature to numerical data.&#x2611; (need to be able to featurize according to image - path to image and text types - transcripts). 
* 'adanet' - Google's AutoML framework in tensorflow (https://github.com/tensorflow/adanet).
* 'alphapy' - keras, scikit-learn, xgboost (https://github.com/ScottfreeLLC/AlphaPy).

Archived modeling techniques:
* 'autokeras' - automatic optimization of a neural network. (https://autokeras.com/) - neural architecture search (takes a very long time). &#x2611; (cannot make predictions from MLP models trained... WTF?)

Note that the autoML techniques are expensive and can take up to 1-2 days to fully train a model.

## Things to finish (before production build)
Other things. 
* If augment == True, can augment (audio, text, image, and video data)
* [recursive feature elimination]() - can help select appropriate features / show feature importances (sc_ script) - use Yellowbrick for this.
* add in [featuretools](https://github.com/Featuretools/featuretools) to create higher-order features to get better accuracy.
* [pLDA](https://github.com/RaviSoji/plda) - implement pLDA for features (help with classification accuracy - dimensionality reduction technique). 
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
