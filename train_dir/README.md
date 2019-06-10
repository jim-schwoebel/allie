## Train directory 

Use this directory to train machine learning models based on folders of files.

Files can take form:

| File type | extension | how to train | 
| ------------- |-------------| -------------| 
| audio file | .WAV, .MP3 | classify multiple folders of .WAV files (e.g. male and female folders) | 
| text file | .WAV, .TXT | classify multiple folders of .TXT files (e.g. wiki vs. transcription) | 
| image file | .PNG, .JPG | classify multiple folders of image files (e.g. dog vs. cat) | 
| video file | .MP4 | classify multiple folders of video files (e.g. walking vs. running) | 
| CSV file | .CSV | classify multiple .CSV files in 2 separate folders (e.g. accepted vs. not accepted as an application candidate). You can input any of the Ludwig data types. (binary, numerical, category, set, bag, sequence, text, timeseries, image) | 

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

## Additional documentation
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)

## Tutorials
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.
* [Age/gender](https://towardsdatascience.com/predict-age-and-gender-using-convolutional-neural-network-and-opencv-fd90390e3ce6) - age and gender detection from images 
* [fft python](https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files)

