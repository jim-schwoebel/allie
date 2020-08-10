## Getting started

Here is a way to quickly augment a folder of text files:
```
cd ~ 
cd allie/features/text_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

### Text
* [augment_textacy](https://github.com/jim-schwoebel/allie/blob/master/augmentation/text_augmentation/augment_textacy.py) - uses [textacy](https://chartbeat-labs.github.io/textacy/build/html/index.html) to augment text files.

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_text_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation) | the default augmentation strategies used during model training for text data if augment_data == True | ["augment_textacy"] | ["augment_textacy", "augment_summary"]  | 
