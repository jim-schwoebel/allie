# voice_modeling
Master repository for modeling voice files. Transformed from NLX-model.

1. Get dataset
2. feature selection (https://machinelearningmastery.com/feature-selection-machine-learning-python/) 
3. modeling (if < 1000 then simple classification, if >1000 deep learning techniques - iterate through various architectures).
4. apply models
5. visualize models (Yellowbrick) - feature selection / etc. 
6. compress models for production 

## types of data

Load folders and data script type based on principal type of file.

* Audio --> .WAV / .MP3 --> .WAV 
* Text --> .TXT / .PPT / .DOCX --> .TXT
* Images --> .PNG / .JPG --> .PNG 
* Video --> .MP4 / .M4A --> .MP4 

## Features to add

### Audio features
* [Prosode Aligner](https://github.com/prosodylab/Prosodylab-Aligner)
* [Parselmouth features](https://github.com/drfeinberg/genderless)
* [DisVoice](https://github.com/jcvasquezc/DisVoice)

### Video features 
* [Semantic video features](https://github.com/JunweiLiang/Semantic_Features)
* [GMM Video features](https://github.com/jonasrothfuss/videofeatures)

### Text features 
* [BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
* [AllenNLP](https://github.com/allenai/allennlp)

### Visualization
* [Yellowbrick]()

### Modeling 
* TPOT
* AutoKeras
* Luwdig  

### Model compression
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor) - should be good for Keras.
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble) - should be good for TPOT I believe.
