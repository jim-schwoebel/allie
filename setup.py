'''
Custom setup script for all operating systems.
'''
import os, json, sys

def brew_install(modules):
  for i in range(len(modules)):
    os.system('brew install %s'%(modules[i]))

# uninstall openCV to guarantee the right version
# this is the only step requiring manual intervention 
# so it's best to do this first.
os.system('pip3 install --upgrade pip -y')
os.system('pip3 uninstall opencv-python -y')
os.system('pip3 uninstall opencv-contrib-python -y')
curdir=os.getcwd()

# possible operating systems
# | Linux (2.x and 3.x) | linux2 (*) |
# | Windows             | win32      |
# | Windows/Cygwin      | cygwin     |
# | Windows/MSYS2       | msys       |
# | Mac OS X            | darwin     |
# | OS/2                | os2        |
# | OS/2 EMX            | os2emx     |

# assumes Mac OSX for SoX and FFmpeg installations
if sys.platform.lower() in ['darwin', 'os2', 'os2emx']:
  brew_modules=['sox', 'ffmpeg', 'opus-tools', 'opus', 'autoconf', 'automake', 'm4', 'libtool', 'gcc', 'portaudio', 'lasound']
  brew_install(brew_modules)
  os.system('pip3 install -r mac.txt')
  # to install opensmile package
  curdir=os.getcwd()
  os.chdir('features/audio_features/helpers/opensmile/opensmile-2.3.0')
  os.system('bash autogen.sh')
  os.system('bash autogen.sh')
  os.system('./configure')
  os.system('make -j4 ; make')
  os.system('make install')
  os.chdir(curdir)
  # install xcode if it is not already installed (on Mac) - important for OPENSMILE features
  os.system('xcode-select --install')
elif sys.platform.lower() in ['linux', 'linux2']:
  os.system('sudo apt-get install ffmpeg -y')
  os.system('sudo apt-get install sox -y')
  os.system('sudo apt-get install python-pyaudio -y')
  os.system('sudo apt-get install portaudio19-dev -y')
  os.system('sudo apt-get install libpq-dev python3.7-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y')
  os.system('sudo apt upgrade gcc -y')
  os.system('sudo apt-get install -y python python-dev python-pip build-essential swig git libpulse-dev')
  os.system('sudo apt-get install -y tesseract-ocr')
  os.system('sudo apt install -y opus-tools')
  os.system('sudo apt install -y libav-tools')
  os.system('sudo apt install -y libsm6')
  # to install opensmile package / link
  os.system('sudo apt-get install autoconf automake m4 libtool gcc -y')
  curdir=os.getcwd()
  os.chdir('features/audio_features/helpers/opensmile/opensmile-2.3.0')
  os.system('bash autogen.sh')
  os.system('bash autogen.sh')
  os.system('./configure')
  os.system('make -j4 ; make')
  os.system('sudo make install')
  os.system('sudo ldconfig')
  os.chdir(curdir)
  os.system('pip3 install -r requirements.txt')

elif sys.platform.lower() in ['win32', 'cygwin', 'msys']: 
  # https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10
  print('you have to install FFmpeg from source') 
  # https://github.com/JoFrhwld/FAVE/wiki/Sox-on-Windows
  print('you have to install SoX from source') 
  # now install all modules with pip3 - install individually to reduce errors
  os.system('pip3 install -r requirements.txt')

# custom installations across all operating systems
os.system('pip3 install git+https://github.com/detly/gammatone.git')
os.system('pip3 install https://github.com/vBaiCai/python-pesq/archive/master.zip')
os.system('pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]')
os.system('pip3 install markovify==0.8.3')
os.system('pip3 install tsaug==0.2.1')
os.system('pip3 install seaborn==0.10.1')
os.system('pip3 install psutil==5.7.2')
os.system('pip3 install pyfiglet==0.8.post1')
os.system('pip3 install gensim==3.8.3')
os.system('pip3 install wget==3.2')
os.system('pip3 install textblob==0.15.3')
os.system('pip3 install moviepy==1.0.3')
os.system('pip3 install textacy==0.10.0')

# install add-ons to NLTK 
os.system('pip3 install nltk==3.4.5')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# install spacy add-ons
os.system('python3 -m spacy download en')
os.system("python3 -m spacy download 'en_core_web_sm'")

# install hyperopt-sklearn
curdir=os.getcwd()
os.chdir(curdir+'/training/helpers/hyperopt-sklearn')
os.system('pip3 install -e .')

# install keras-compressor
os.chdir(curdir)
os.chdir(curdir+'/training/helpers/keras_compressor')
os.system('pip3 install .')

# go back to host directory
os.chdir(curdir)

# now go setup tests
os.chdir('tests')
os.system('python3 test.py')