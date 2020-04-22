'''
@package: pyAudioLex
@author: Drew Morris
@module: pronoun_to_noun_ratio

Pronoun-to-noun ratio is the ratio of the total number of pronouns to 
the total number of nouns.
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

def pronoun_to_noun_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  pos = pos_tag(tokens)
  pronouns = []
  nouns = []

  for [token, tag] in pos:
    part = map_tag('en-ptb', 'universal', tag)
    if part == "PRON":
      pronouns.append(token)

  for [token, tag] in pos:
    part = map_tag('en-ptb', 'universal', tag)
    if part == "NOUN":
      nouns.append(token)

  if len(nouns) == 0:
    return float(0)
  else:
    return float(len(pronouns)) / float(len(nouns))
