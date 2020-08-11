'''
Custom setup script for all operating systems.
'''
import os, json, sys, nltk

# add-on script for docker
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# now go setup tests
os.chdir('tests')
os.system('python3 test.py')