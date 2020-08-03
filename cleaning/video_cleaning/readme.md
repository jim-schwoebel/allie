### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| [default_video_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning) | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
