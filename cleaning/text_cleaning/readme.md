### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| [default_text_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/text_cleaning) | the default cleaning techniques used during model training on text data if clean_data == True| ["clean_textacy"] | ["clean_summary", "clean_textacy"]  | 
