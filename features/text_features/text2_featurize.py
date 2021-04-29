import nltk
nltk.download('universal_tagset')

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import numpy as np
import math

def filler_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('uh|ugh|um|like|you know')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

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

def entropy(tokens):
  freqdist = FreqDist(tokens)
  probs = [freqdist.freq(l) for l in freqdist]

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

def question_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('Who|What|When|Where|Why|How|\?')
  qtokens = tokenizer.tokenize(s)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

def number_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  tokenizer = RegexpTokenizer('zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|couple|several|few|\d')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

def brunets_index(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  N = float(len(tokens))
  V = float(len(set(tokens)))
  
  if N == 0 or V == 0:
    return float(0)
  else:
    return math.pow(N, math.pow(V, -0.165))

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


def wpm(s, tokens, duration):
  r = float(duration / 60)

  return len(tokens) / r

def text2_featurize(transcript):
    features=[filler_ratio(transcript),
              type_token_ratio(transcript),
              standardized_word_entropy(transcript),
              question_ratio(transcript),
              number_ratio(transcript),
              brunets_index(transcript),
              honores_statistic(transcript),
              pronoun_to_noun_ratio(transcript)]
    labels=['filler_ratio', 'type_token_ratio', 'standardized_word_entropy',
            'question_ratio', 'number_ratio', 'brunets_index', 'honores_statistic',
            'pronoun_to_noun_ratio']

    return features, labels

# features, labels =text2_featurize('this is a test transcript.')
# print(features)
# print(labels)