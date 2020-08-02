## How to annotate

You can simply annotate using the command-line interface here:

```python3
python3 annotate.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

Converting to csv
```
python3 create_csv.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

## Future work

## with timesplit

- Audio - https://github.com/jim-schwoebel/sound_event_detection/tree/94da2fe402ef330e0b6dc9ed41b59b0902e67842 (audio frames, text transcription frames)
- Text - https://github.com/doccano/doccano
- Image - https://github.com/tzutalin/labelImg/tree/c1c1dbef315df52daad9b22a418c2e832b60dae5
- Video - https://github.com/ElbitSystems/AnnotationTool (audio, image(s), and text transcription frames) and https://github.com/xinshuoweng/AB3DMOT (3D object tracking)
- [submodule "datasets/labeling/sound_event_detection"] path = datasets/labeling/sound_event_detection url = https://github.com/jim-schwoebel/sound_event_detection 
- [submodule "datasets/labeling/labelImg"] path = datasets/labeling/labelImg url = https://github.com/tzutalin/labelImg
- [submodule "datasets/labeling/labelme"] path = datasets/labeling/labelme url = https://github.com/wkentaro/labelm

### things to do
- classification and/or regression problem (if regression, float, if classification, integer value)
- add classnames (to Optparser)
- add ability to split up each file into more files in the folder {audio file / video file --> audio file 1, 2, n...} 
