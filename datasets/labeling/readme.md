# Labeling

Allie also has data labeling capabilities.

Data labeling is the process of tagging data with useful information. For example, determining whether a speaker is a male or a female.

This allows for data to be modeled as classification problems (labels are binary or [1,2,3,4,...N]) or regression problems (labels are [0:1]). 

## Manual labeling 

Manual labeling involves a human-in-the-loop. A human being manually looks at or listens to a dataset to tag it with useful information. For example, a human may view a 20 second video clip and rank each emotion - happy, sad, neutral, angry, disgust - on a scale of 1-10. This can then be useful to build regression or classification-based machine learning models.

There are many tools for manual annotation. Here are the ones that we have found most useful internally. 
* [sound_event_detection](https://github.com/jim-schwoebel/sound_event_detection/tree/94da2fe402ef330e0b6dc9ed41b59b0902e67842) - for audio 
* [labelme](https://github.com/wkentaro/labelme/tree/a98d9b66b032622685c8d59c7712be37eef9d3e5) - for images
* [labellmg](https://github.com/tzutalin/labelImg/tree/c1c1dbef315df52daad9b22a418c2e832b60dae5) - for images/videos
* [youtube_scrape](https://github.com/jim-schwoebel/allie/tree/master/datasets/labeling/youtube_scrape) - for manually annotating YouTube videos via a spreadsheet / building playlists.

## Automatic labeling

Automatic labeling is the idea of using machine learning models to automatically label datasets with useful information. For example, you may want to know in general what the age, gender, and ethnicity of a voice dataset. 

Allie can auto-label if you use machine learning models in the load_dir. 

Read more information about how to do this [here](https://github.com/jim-schwoebel/allie/tree/master/models). 
