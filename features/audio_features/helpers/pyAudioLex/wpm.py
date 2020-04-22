'''
@package: pyAudioLex
@author: Drew Morris
@module: wpm

Used to calculate words per minute.
'''

def wpm(s, tokens, duration):
  r = float(duration / 60)

  return len(tokens) / r