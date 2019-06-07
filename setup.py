'''
Custom setup script for MacOSX.
'''
import os 

def brew_install(modules):
  for i in range(len(modules)):
    os.system('brew install %s'%(modules[i]))
    
brew_modules=['sox']
brew_install(brew_modules)
os.system('pip3 install -r requirements.txt')
