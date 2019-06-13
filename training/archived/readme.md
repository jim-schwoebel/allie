## Archived modeling techniques 

I tried a wide range of autoML libraries. These did not make the cut for these reasons. 

* '[autokeras](https://autokeras.com/)' - automatic optimization of a neural network using neural architecture search (takes a very long time). (cannot make predictions from MLP models trained... WTF?) 👎 
* '[autosklearn](https://github.com/automl/auto-sklearn)' - added to pip3 installation + script. (segmentation faults are common, thus archived. If documentation and community improves, may be good to add back in). 👎 
* '[pLDA](https://github.com/RaviSoji/plda)' - this works only for symmetrical images (as it cannot compute eigenvector for many of the feature arrays we have created). For this reason, it is probably not a good idea to use this as a standard training method. 👎 

Note that as the documentation and support for these libraries becomes better over time, it may make sense to move them into production.

They just don't work now :-) 
