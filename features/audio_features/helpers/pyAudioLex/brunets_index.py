'''
@package: pyAudioLex
@author: Drew Morris
@module: brunets_index

Brunet's index (W) quantifies lexical richness [20]. It is 
calculated as W = N^V^-0.165, where N is the total text length and V is the 
total vocabulary. Lower values of W correspond to richer texts. As with 
standardized word entropy, stemming is done on words and only the stems 
are considered.
'''

import math
from nltk.tokenize import word_tokenize

def brunets_index(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  N = float(len(tokens))
  V = float(len(set(tokens)))
  
  if N == 0 or V == 0:
    return float(0)
  else:
    return math.pow(N, math.pow(V, -0.165))
