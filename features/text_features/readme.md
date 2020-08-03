## How to use
```
cd allie/features/text_features
python3 featurize.py [folder] [featuretype]
```

### Text
* [bert features](https://github.com/UKPLab/sentence-transformers) - extract BERT-related features from sentences (note shorter sentences run faster here, and long text can lead to long featurization times).
* [fast_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/fast_features.py)
* [glove_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/glove_features.py)
* [grammar_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/grammar_features.py) - 85k+ grammar features (memory intensive)
* [nltk_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/nltk_features.py) - standard text feature array (default)
* [spacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/spacy_features.py)
* [textacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/textacy_features.py) - a variety of document classification and topic modeling features
* [text_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/text_features.py) - many different types of features like emotional word counts, total word counts, Honore's statistic and others.
* [w2v_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/w2vec_features.py) - note this is the largest model from Google and may crash your computer if you don't have enough memory. I'd recommend fast_features if you're looking for a pre-trained embedding.
