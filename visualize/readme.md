## Visualization

This is a univeral visualizer for all types of data. 

Note that this is a setting in the Allie Framework (e.g. visualization == True | False).

## Getting started
To get started, you first need to featurize some data using featurizations scripts. This data must be in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir) in the form of directories. To read more about featurization, [see this page](https://github.com/jim-schwoebel/allie/tree/master/features).

After you have featurized your data, go to this current folder (./allie/visualize') and run the visualize.py script:
```
python3 visualize.py [problemtype] [folder A] [folder B] ... [folder N]
```
Note you need to pass through the problem_type (e.g. 'audio'|'text'|'image'|'video'|'csv') and also all the relevant folders featurizations. In this case, we are looking at audio files separating males from females (e.g. featurizations exist) in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir).
```
python3 visualize.py audio males females 
```

## Output graphs

Once you run this script, you output many visualizations. These visualizations can be customized within the script itself with some simple modifications. See below for some of the visualizations you can make.

### numbers in each class
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/classes.png)

## top 10 features (if 10 features exist)
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_importance.png)

## PCA plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/pca.png)

## Pearson ranking plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/pearson.png)

## tSNE plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/tsne.png)

## Using streamlit

```
streamlit run test.py
```
Outputs various graphs on the data.

## Featurization tutorials (Readmes)
* [Feature selection techniques in Python](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)
* [How to visualize anything in Yellowbrick](https://medium.com/analytics-vidhya/how-to-visualize-anything-in-machine-learning-using-yellowbrick-and-mlxtend-39c45e1e9e9f)
* [Intro to data visualization in Python](https://gilberttanner.com/blog/introduction-to-data-visualization-inpython)
* [Matplotlib bar chart](https://pythonspot.com/matplotlib-bar-chart/)
* [Visualize machine learning Pandas](https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/)
* [Top 50 Matplotlib visualizations](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/)
* [Dimensionality reduction techniques](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)

## Additional documentation
* [MLXtend](https://github.com/rasbt/mlxtend) - for visualizing models
* [Plotly learning curves](https://github.com/mitmedialab/vizml/blob/master/notebooks/Plotly%20Performance.ipynb)
* [Snowflake](https://github.com/doubledherin/Audio_Snowflake) - audio song visualization
* [Streamit](https://github.com/streamlit/streamlit)
* [Voicebook: Chapter 6 Visualization](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_6_visualization)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)
