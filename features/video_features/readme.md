## How to use
```
cd allie/features/video_features
python3 featurize.py [folder] [featuretype]
```

### Videos 
* [video_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/video_features.py) - standard video feature array (default)
* [y8m_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/y8m_features.py) 

### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| default_video_transcriber | the default transcription technique used for videos (.mp4 --> text from the video) | ["tesseract (averaged over frames)"] | ["tesseract (averaged over frames)"] |
| transcribe_video | a setting to define whether or not to transcribe video files during featurization and model training via the default_video_transcriber | True | True, False | 
