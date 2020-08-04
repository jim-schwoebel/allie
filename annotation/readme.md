## How to annotate

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/label.png)

You can simply annotate using the command-line interface here:

```python3
cd /Users/jim/desktop/allie
cd annotation
python3 annotate.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

What results is annotated folders in .JSON format following the [standard dictionary](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py).

```python3
{"sampletype": "audio", "transcripts": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "features": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "models": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "labels": [{"male": {"value": 1.0, "datetime": "2020-08-03 14:06:53.101763", "filetype": "audio", "file": "1.wav", "problemtype": "classification", "annotate_dir": "/Users/jim/desktop/allie/train_dir/males"}}], "errors": [], "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}}
```

After you annotate, you can create [a nicely formatted .CSV](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/male_data.csv) for machine learning:

```python3
cd /Users/jim/desktop/allie
cd annotation
python3 create_csv.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

Click the .GIF below for a quick tutorial and example.

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/annotation.gif)](https://drive.google.com/file/d/1Xn7A61XWY8oCAfMmjSMpwEjvItiNp5ev/view?usp=sharing)

## What each CLI argument represents
| CLI argument | description | possible options | example |
|------|------|------|------|
| -d | the specified directory full of files that you want to annotate. | any directory path | "/Users/jim/desktop/allie/train_dir/males/" |
| -s | the file type / type of problem that you are annotating | ["audio", "text", "image", "video"] | "audio" | 
| -c | the class name that you are annotating | any string / class name | "male" | 
| -p | the type of problem you are annotating - classification or regression label | ["classification", "regression"] | "classification" | 

## Other annotation methods

### Audio 
* [sound_event_detection](https://github.com/jim-schwoebel/sound_event_detection/tree/94da2fe402ef330e0b6dc9ed41b59b0902e67842)
### Text 
* [doccano](https://github.com/doccano/doccano)
### Image 
* [labelImg](https://github.com/tzutalin/labelImg/tree/c1c1dbef315df52daad9b22a418c2e832b60dae5) 
### Video
* [AnnotationTool](https://github.com/ElbitSystems/AnnotationTool)
* [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
