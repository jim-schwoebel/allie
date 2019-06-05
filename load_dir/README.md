# load_dir

Use this directory to make predictions on audio, image, video, or text files.

Specifically, just drag and drop sample data in here and predictions will be made based on the models in the ./models directory.

## detailed instructions (classification predictions) 

1. Create a list of directories of classes
2. Put relevant files in the folders for the classes (e.g. audio, .txt, image, video, or .CSV files)
3. Run load_classify.py (runs all models in ./models/ directory + makes predictions) 
4. See predictions for each file in a .JSON format. The feature array will follow the standard feature array vector along with any audio, image, or video transcripts that were made. 

## detailed instructions (regression predictions)  

1. convert to .CSV file format 
2. run load_regression script (TPOT)
