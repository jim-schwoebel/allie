import sys
sys.path.append('..')
import wave
import contextlib
import glob
import json
import pyAudioLex
from colorama import Fore, Back, Style

default_format = '.3f'

# question_ratio
question_ratio = pyAudioLex.question_ratio('What time is it? What day is it?')
print('question_ratio', format(question_ratio, default_format))
assert format(question_ratio, default_format) == '0.400'
assert format(pyAudioLex.question_ratio(''), default_format) == '0.000'

# filler_ratio
filler_ratio = pyAudioLex.filler_ratio('Uh I am, like, hello, you know, and um or umm.')
print('filler_ratio', format(filler_ratio, default_format))
assert format(filler_ratio, default_format) == '0.312'
assert format(pyAudioLex.filler_ratio(''), default_format) == '0.000'

# verb_freq
verb_freq = pyAudioLex.verb_freq('They refuse to permit us to obtain the refuse permit.')
print('verb_freq', format(verb_freq, default_format))
assert format(verb_freq, default_format) == '0.273'
assert format(pyAudioLex.verb_freq(''), default_format) == '0.000'

# noun_freq
noun_freq = pyAudioLex.noun_freq('They refuse to permit us to obtain the refuse permit.')
print('noun_freq', format(noun_freq, default_format))
assert format(noun_freq, default_format) == '0.182'
assert format(pyAudioLex.noun_freq(''), default_format) == '0.000'

# pronoun_freq
pronoun_freq = pyAudioLex.pronoun_freq('They refuse to permit us to obtain the refuse permit.')
print('pronoun_freq', format(pronoun_freq, default_format))
assert format(pronoun_freq, default_format) == '0.182'
assert format(pyAudioLex.pronoun_freq(''), default_format) == '0.000'

# adverb_freq
adverb_freq = pyAudioLex.adverb_freq('They refuse to permit us to obtain the refuse permit.')
print('adverb_freq', format(adverb_freq, default_format))
assert format(adverb_freq, default_format) == '0.000'
assert format(pyAudioLex.adverb_freq(''), default_format) == '0.000'

# adjective_freq
adjective_freq = pyAudioLex.adjective_freq('They refuse to permit us to obtain the refuse permit.')
print('adjective_freq', format(adjective_freq, default_format))
assert format(adjective_freq, default_format) == '0.000'
assert format(pyAudioLex.adjective_freq(''), default_format) == '0.000'

# particle_freq
particle_freq = pyAudioLex.particle_freq('They refuse to permit us to obtain the refuse permit.')
print('particle_freq', format(particle_freq, default_format))
assert format(particle_freq, default_format) == '0.182'
assert format(pyAudioLex.particle_freq(''), default_format) == '0.000'

# conjunction_freq
conjunction_freq = pyAudioLex.conjunction_freq('They refuse to permit us to obtain the refuse permit.')
print('conjunction_freq', format(conjunction_freq, default_format))
assert format(conjunction_freq, default_format) == '0.000'
assert format(pyAudioLex.conjunction_freq(''), default_format) == '0.000'

# pronoun_to_noun_ratio
pronoun_to_noun_ratio = pyAudioLex.pronoun_to_noun_ratio('They refuse to permit us to obtain the refuse permit.')
print('pronoun_to_noun_ratio', format(pronoun_to_noun_ratio, default_format))
assert format(pronoun_to_noun_ratio, default_format) == '1.000'
assert format(pyAudioLex.pronoun_to_noun_ratio(''), default_format) == '0.000'

# standardized_word_entropy
standardized_word_entropy = pyAudioLex.standardized_word_entropy('male female male female')
print('standardized_word_entropy', format(standardized_word_entropy, default_format))
assert format(standardized_word_entropy, default_format) == '0.721'
assert format(pyAudioLex.standardized_word_entropy(''), default_format) == '0.000'

# number_ratio
number_ratio = pyAudioLex.number_ratio('I found seven apples by a couple of trees. I found a dozen eggs by those 5 chickens.')
print('number_ratio', format(number_ratio, default_format))
assert format(number_ratio, default_format) == '0.200'
assert format(pyAudioLex.number_ratio(''), default_format) == '0.000'

# brunets_index
brunets_index = pyAudioLex.brunets_index('Bravely bold Sir Robin, rode forth from Camelot.')
print('brunets_index', format(brunets_index, default_format))
assert format(brunets_index, default_format) == '4.830'
assert format(pyAudioLex.brunets_index(''), default_format) == '0.000'

# honores_statistic
honores_statistic = pyAudioLex.honores_statistic('Bravely bold Sir Robin, rode forth from Camelot. Afterwards, Sir Robin went to the castle.')
print('honores_statistic', format(honores_statistic, default_format))
assert format(honores_statistic, default_format) == '1104.165'
assert format(pyAudioLex.honores_statistic(''), default_format) == '0.000'

# type_token_ratio
type_token_ratio = pyAudioLex.type_token_ratio('Bravely bold Sir Robin, rode forth from Camelot. Afterwards, Sir Robin went to the castle.')
print('type_token_ratio', format(type_token_ratio, default_format))
assert format(type_token_ratio, default_format) == '0.579'
assert format(pyAudioLex.type_token_ratio(''), default_format) == '0.000'

# ---------------------
# test the audio
print 'Processing test audio sample...'
recording_id = 'NLX-10'
samplejson = './test/process_inq/' + recording_id + '.json'
samplewav = './test/process_inq/' + recording_id + '.wav'

# get json
jsonfile = open(samplejson, 'r')
data = json.loads(jsonfile.read())
transcript = data['transcript'].replace('[','').replace('?]','')

# get wav
wavefile = wave.open(samplewav, 'r')

# see how long the file is
with contextlib.closing(wavefile) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

features = pyAudioLex.process(transcript, duration, samplewav)

brunets_index = features['linguistic']['brunets_index']
print('asserting linguistic feature', format(brunets_index, default_format))
assert format(brunets_index, default_format) == '11.844'

zcr = features['audio']['ZCR'][2]
print('asserting audio feature', format(zcr, default_format))
assert format(zcr, default_format) == '0.021'

print 'Done!'
