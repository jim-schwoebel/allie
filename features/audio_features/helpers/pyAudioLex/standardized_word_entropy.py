'''
@package: pyAudioLex
@author: Drew Morris
@module: standardized_word_entropy

One of the earliest parts of the brain to be damaged by Alzheimer's 
disease is the part of the brain that deals with language ability [5]. 
We hypothesize that this may cause a degradation in the variety of words 
and word combinations that a patient uses. Standardized word entropy, 
i.e., word entropy divided by the log of the total word count, is used 
to model this phenomenon. Because the aim is to compute the variety of word 
choice, stemming is done, and only the stems of the words are considered.
'''

import math
from nltk import FreqDist
from nltk.tokenize import word_tokenize

def entropy(tokens):
  freqdist = FreqDist(tokens)
  probs    = [freqdist.freq(l) for l in freqdist]

  return -sum(p * math.log(p, 2) for p in probs)

def standardized_word_entropy(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  if len(tokens) == 0:
    return float(0)
  else:
    if math.log(len(tokens)) == 0:
      return float(0)
    else:
      return entropy(tokens) / math.log(len(tokens))
