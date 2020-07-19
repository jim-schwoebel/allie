'''
Quick installation script to get everything up-and-running.
'''
import os

os.system('brew install opus-tools')
os.system('brew install opus')
os.system('brew install sox')
os.system('brew install ffmpeg')
os.system('pip3 install -U nltk')
import nltk
nltk.download('wordnet')
os.system('pip3 install -r requirements.txt')
