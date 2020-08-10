## Getting started

Here is a way to quickly augment a folder of csv files:
```
cd ~ 
cd allie/features/csv_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

### [CSV](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation)
* [augment_tgan_classification](https://github.com/sdv-dev/TGAN) - generative adverserial examples - can be done on class targets / problems.
* [augment_ctgan_regression]() - generative adverserial example on regression problems / targets.

## Settings
| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_csv_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation) | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification", "augment_ctgan_regression"]  | 
