'''
@package: pyAudioLex
@author: Drew Morris
@module: conjunction_freq

Frequency of a POS tag is computed by dividing the total number of words 
with that tag by the total number of words spoken by the subject in the 
recording.
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

def conjunction_freq(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  pos = pos_tag(tokens)
  conjunctions = []

  for [token, tag] in pos:
    part = map_tag('en-ptb', 'universal', tag)
    if part == "CONJ":
      conjunctions.append(token)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(conjunctions)) / float(len(tokens))
