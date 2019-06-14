# load_dir

Use this directory to make predictions on audio, image, video, or text files.

Specifically, just drag and drop sample data in here and predictions will be made based on the models in the ./models directory.

## load by model type
* **'tpot'** - TPOT classification / regression (autoML). &#x2611;
* 'scsr' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* 'plda' - probabilistic LDA modeling. &#x2611; (unstable)
* 'keras' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* 'autokeras' - automatic optimization of a neural network. (https://autokeras.com/) - neural architecture search (takes a very long time). &#x2611;
* 'devol' - genetic programming keras (https://github.com/joeddav/devol.git). &#x2611;
* 'ludwig' - deep learning (simple ludwig). - convert every feature to numerical data.&#x2611;

## settings
* if compress == True, load in compressed model into memory and make predictions 
* load all models by folder and model type (audio_models, text_models, etc.)
* only load models in if there are data in the folder. 

## detailed instructions (classification predictions) 

1. Create a list of directories of classes
2. Put relevant files in the folders for the classes (e.g. audio, .txt, image, video, or .CSV files)
3. Run load_classify.py (runs all models in ./models/ directory + makes predictions) 
4. See predictions for each file in a .JSON format. The feature array will follow the standard feature array vector along with any audio, image, or video transcripts that were made. 

## Future additions
* [Age/gender](https://towardsdatascience.com/predict-age-and-gender-using-convolutional-neural-network-and-opencv-fd90390e3ce6) - age and gender detection from images 
