'''
This is from the BERT Model

https://github.com/UKPLab/sentence-transformers

'''
from sentence_transformers import SentenceTransformer
import numpy as np

def bert_featurize(sentence,model):
	features = model.encode(sentence)
	labels=list()
	for i in range(len(features)):
		labels.append('bert_feature_%s'%(str(i+1)))

	return features, labels

