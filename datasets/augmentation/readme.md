## Augmentation 

This part of Allie's skills relates to data augmentation.

Data augmentation is used to expand the training dataset in order to improve the performance and ability of a machine learning model to generalize. For example, you may want to shift, flip, brightness, and zoom on images to augment datasets to make models perform better in noisy environments indicative of real-world use. Data augmentation is especially useful when you don't have that much data, as it can greatly expand the amount of training data that you have for machine learning. 

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

You can read more about data augmentation [here](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c).

## Augmentation scripts 

```
augment.py [foldername]
```

Folder structures 
- audio_augmentation
- text_augmentation 
- image_augmentation
- video_augmentation 
- csv_augmentation 

## Work-in-progress (WIP)



## References
* [1000x Faster Data Augmentation](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c)
- [AutoAugment (Google)](https://github.com/tensorflow/models/tree/master/research/autoaugment) - paper [here](https://arxiv.org/abs/1805.09501)
- [Population Based Augmentation (PBA)](https://github.com/arcelien/pba) - paper [here](https://arxiv.org/abs/1711.09846
)
