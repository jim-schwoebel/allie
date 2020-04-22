'''
@package: pyAudioLex
@author: Drew Morris
@module: honores_statistic

Honore's statistic [21] is based on the notion that the larger the number 
of words used by a speaker that occur only once, the richer his overall 
lexicon is. Words spoken only once (V1) and the total vocabulary used (V) 
have been shown to be linearly associated. Honore's statistic generates a 
lexical richness measure according to R = (100 x log(N)) / (1 _ (V1 / V)), 
where N is the total text length. Higher values correspond to a richer 
vocabulary. As with standardized word entropy, stemming is done on words 
and only the stems are considered.
'''

import math
from nltk.tokenize import word_tokenize
from nltk import FreqDist

def honores_statistic(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  uniques = []

  for token, count in FreqDist(tokens).items():
    if count == 1:
      uniques.append(token)

  N  = float(len(tokens))
  V  = float(len(set(tokens)))
  V1 = float(len(uniques))
    
  if N == 0 or V == 0 or V1 == 0:
    return float(0)
  elif V == V1:
    return (100 * math.log(N))
  else:
    return (100 * math.log(N)) / (1 - (V1 / V))
