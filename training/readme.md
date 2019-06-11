## Settings

Training scripts here. 5 types
* 'tpot' - TPOT classification / regression (autoML).
* 'scsr' - simple classification / regression.
* 'ludwig' - deep learning (simple ludwig).
* 'keras' - automatic optimization of a neural network.
* 'plda' - probabilistic LDA modeling.

## Things to do 

* If augment == True, can augment (audio, text, image, and video data)
* [recursive feature elimination]() - can help select appropriate features / show feature importances (sc_ script) - use Yellowbrick for this.
* [pLDA](https://github.com/RaviSoji/plda) - implement pLDA for features (help with classification accuracy - dimensionality reduction technique). 

## Data modeling tutorials (Readmes)
* Audio file training example
* Text file training example 
* Image file training example
* Video file training example 
* CSV file training example

## Tutorials
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.
* [Age/gender](https://towardsdatascience.com/predict-age-and-gender-using-convolutional-neural-network-and-opencv-fd90390e3ce6) - age and gender detection from images 
* [fft python](https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files)

## Additional Settings
### 1. Compress models? 
--> if True, compress with AutoKeras or scikit-learn compress (faster predictions from features) 
### 2. Create YAML file (for GitHub repository) - if True, select a cool .GIF to put in repo readme 
* docker container 
* featurizers
* automated testing (test file) 
* ...

## Additional documentation
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)
