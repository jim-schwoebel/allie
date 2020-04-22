'''
@package: pyAudioLex
@author: Drew Morris
@module: type_token_ratio

A pattern that we noticed in the recordings of the Alzheimer's 
patients is the frequency of repetitions in conversation. Patients tend 
to forget what they have said and to repeat it elsewhere in the 
conversation. The metric that we used to measure this phenomenon is 
type-token ratio [22]. Type-token ratio is defined as the ratio of 
the number of unique words to the total number of words. In order to 
better assess the repetitions, only the stems of the words are considered 
in calculations.
'''

from nltk.tokenize import word_tokenize
from nltk import FreqDist

def type_token_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  uniques = []

  for token, count in FreqDist(tokens).items():
    if count == 1:
      uniques.append(token)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(uniques)) / float(len(tokens))
