## How to annotate

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/label.png)

You can simply annotate using the command-line interface here:

```python3
python3 annotate.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

After you annotate, you can create a nicely formatted .CSV for machine learning:

```python3
python3 create_csv.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

Click the .GIF below for a quick tutorial and example.

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/annotation.gif)](https://drive.google.com/file/d/1Xn7A61XWY8oCAfMmjSMpwEjvItiNp5ev/view?usp=sharing)

## What each CLI argument represents
| CLI argument | description | options | example |
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
