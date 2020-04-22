'''
@package: pyAudioLex
@author: Drew Morris
@module: pyAudioLex

The main library API
'''

from .audio import audio, get_duration
from nltk.tokenize import word_tokenize
from .question_ratio import question_ratio
from .filler_ratio import filler_ratio
from .verb_freq import verb_freq
from .noun_freq import noun_freq
from .pronoun_freq import pronoun_freq
from .adverb_freq import adverb_freq
from .adjective_freq import adjective_freq
from .particle_freq import particle_freq
from .conjunction_freq import conjunction_freq
from .pronoun_to_noun_ratio import pronoun_to_noun_ratio
from .standardized_word_entropy import standardized_word_entropy
from .number_ratio import number_ratio
from .brunets_index import brunets_index
from .honores_statistic import honores_statistic
from .type_token_ratio import type_token_ratio
from .wpm import wpm

# get the duration
def process_duration(wav):
  return get_duration(wav)

# get linguistic features
def process_linguistic(s='', duration=0.0):
  tokens  = word_tokenize(s)
  features = {}
  features['question_ratio'] = question_ratio(s, tokens)
  features['filler_ratio'] = filler_ratio(s, tokens)
  features['verb_freq'] = verb_freq(s, tokens)
  features['noun_freq'] = noun_freq(s, tokens)
  features['pronoun_freq'] = pronoun_freq(s, tokens)
  features['adverb_freq'] = adverb_freq(s, tokens)
  features['adjective_freq'] = adjective_freq(s, tokens)
  features['particle_freq'] = particle_freq(s, tokens)
  features['conjunction_freq'] = conjunction_freq(s, tokens)
  features['pronoun_to_noun_ratio'] = pronoun_to_noun_ratio(s, tokens)
  features['standardized_word_entropy'] = standardized_word_entropy(s, tokens)
  features['number_ratio'] = number_ratio(s, tokens)
  features['brunets_index'] = brunets_index(s, tokens)
  features['honores_statistic'] = honores_statistic(s, tokens)
  features['type_token_ratio'] = type_token_ratio(s, tokens)
  features['wpm'] = wpm(s, tokens, duration)

  return features

# get audio features
def process_audio(wav):
  duration = process_duration(wav)

  # put the results into one object
  features = {}

  # only do features if we have enough duration
  if duration > 0.0:
    features = audio(wav)

  # set duration regardless
  features['duration'] = duration

  return features

# get both if we can
def process(wav, s=''):
  audio = process_audio(wav)
  linguistic = {}

  # only do linguistic features if we have a transcript
  if len(s) >= 1:
    linguistic = process_linguistic(s, audio['duration'])
  
  # put the results into one object
  results = {}
  results['linguistic'] = linguistic
  results['audio'] = audio

  return results
