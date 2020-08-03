
## Getting started
To clean an entire folder of a certain file type (e.g. audio files of .PNG format), you can run:

```
cd ~ 
cd allie/cleaning/image_cleaning
python3 cleaning.py /Users/jimschwoebel/allie/load_dir
```

### Implemented for all file types 
* [delete_duplicates](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_duplicates.py) - deletes duplicate files in the directory 
* [delete_json](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_json.py) - deletes all .JSON files in the directory (this is to clean the featurizations) 

### [Image](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning)
* [clean_extractfaces]() - extract faces from an image
* [clean_greyscale]() - make all images greyscale 
* [clean_jpg2png]() - make images from jpg to png to standardize image formats

### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| [default_image_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning) | the default cleaning techniques used for image data as a part of model training is clean_data == True| ["clean_greyscale"] |["clean_extractfaces", "clean_greyscale", "clean_jpg2png"] | 
