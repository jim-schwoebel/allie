
## Getting started
To clean an entire folder of video files in .MP4 format, you can run:

```
cd ~ 
cd allie/cleaning/audio_cleaning
python3 cleaning.py /Users/jimschwoebel/allie/load_dir
```

### [Video](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning)
* [clean_alignfaces](https://github.com/jim-schwoebel/allie/blob/master/cleaning/video_cleaning/clean_alignfaces.py) - takes out faces from a video frame and keeps the video for an added label
* [clean_videostabilize](https://github.com/jim-schwoebel/allie/blob/master/cleaning/video_cleaning/clean_videostabilize.py) - stabilizes a video frame using [vidgear](https://github.com/abhiTronix/vidgear) (note this is a WIP)

### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| [default_video_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning) | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
