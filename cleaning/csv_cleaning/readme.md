
## Getting started
To clean an entire folder of .CSV files, you can run:

```
cd ~ 
cd allie/cleaning/csv_cleaning
python3 cleaning.py /Users/jimschwoebel/allie/load_dir
```

### Implemented for all file types 
* [delete_duplicates](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_duplicates.py) - deletes duplicate files in the directory 
* [delete_json](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_json.py) - deletes all .JSON files in the directory (this is to clean the featurizations) 

### [CSV](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning)
* [clean_csv]() - uses [datacleaner](https://github.com/rhiever/datacleaner), a standard excel sheet cleaning script that imputes missing values and prepares CSV spreadsheets for machine learning

### Settings

Here are some default settings relevant to this section of Allie's API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | | 
| [default_csv_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning) | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv"] | 
