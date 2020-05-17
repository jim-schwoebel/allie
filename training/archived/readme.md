## Archived modeling techniques 

I tried a wide range of autoML libraries. These did not make the cut for these reasons. 

* '[autobazaar](https://github.com/HDI-Project/AutoBazaar)' - AutoBazaar: An AutoML System from the Machine Learning Bazaar (from the Data to AI Lab at MIT).&#x2611; - this was a bit hard to extract a machine learning model from and the schema was a little hard to make interoperable with Allie; may get to this later if the framework is further supported. 👎 
* '[autokeras](https://autokeras.com/)' - automatic optimization of a neural network using neural architecture search (takes a very long time) - consistently has problems associated with saving and loading models in keras. 👎 
* '[autosklearn](https://github.com/automl/auto-sklearn)' - added to pip3 installation + script. (segmentation faults are common, thus archived. If documentation and community improves, may be good to add back in). 👎 
* '[pLDA](https://github.com/RaviSoji/plda)' - this works only for symmetrical images (as it cannot compute eigenvector for many of the feature arrays we have created). For this reason, it is probably not a good idea to use this as a standard training method. 👎 

Note that as the documentation and support for these libraries becomes better over time, it may make sense to move them into production.

They just don't work now :-) 

## Other (nice-to-have) things to incorporate

* [recursive feature elimination]() - can help select appropriate features / show feature importances (sc_ script) - use Yellowbrick for this.
* add in [featuretools](https://github.com/Featuretools/featuretools) to create higher-order features to get better accuracy.
* hyperparameter optimization - https://github.com/autonomio/talos
* Clustering algorithms 
