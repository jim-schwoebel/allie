# the case of absolute metrics
''' 
taken from https://github.com/aliutkus/speechmetrics

Note there are no references with these metrics.
'''
import speechmetrics

def speechmetrics_featurize(wavfile):
    window_length = 5 # seconds
    metrics = speechmetrics.load('absolute', window_length)
    scores = metrics('test.wav')
    scores['mosnet'] = float(scores['mosnet'])
    scores['srmr'] = float(scores['srmr'])
    features = list(scores.values())
    labels = list(scores)
    return features, labels

# features, labels = speechmetrics_featurize('test.wav')
# print(features)
# print(labels)
