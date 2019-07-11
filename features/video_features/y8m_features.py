import os, sys, tarfile, numpy
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import numpy
import cv2, os, random, json, sys, getpass, pickle, datetime, time, librosa, shutil, gensim, nltk
from nltk import word_tokenize 
from nltk.classify import apply_features, SklearnClassifier, maxent
import speech_recognition as sr
from pydub import AudioSegment
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from operator import itemgetter
from matplotlib import pyplot as plt
from PIL import Image
import skvideo.io
import skvideo.motion
import skvideo.measure
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
from pydub import AudioSegment

def prev_dir(directory):
  g=directory.split('/')
  # print(g)
  lastdir=g[len(g)-1]
  i1=directory.find(lastdir)
  directory=directory[0:i1]
  return directory

# import custom audioset directory 
basedir=os.getcwd() 
prevdir=prev_dir(basedir)
audioset_dir=prevdir+'audio_features'
sys.path.append(audioset_dir)
import audioset_features as af 
print('imported audioset features!')
os.chdir(basedir)

#### to extract tesseract features 
sys.path.append(prevdir+ '/image_features')
import tesseract_features as tff
os.chdir(basedir)

# import fast featurize
text_dir=prevdir+'text_features'
os.chdir(text_dir)
sys.path.append(text_dir)
import fast_features as ff 
print('imported fast features!')
os.chdir(basedir)

INCEPTION_TF_GRAPH = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
YT8M_PCA_MAT = 'http://data.yt8m.org/yt8m_pca.tgz'
MODEL_DIR = os.path.join(os.getenv('HOME'), 'yt8m')

class YouTube8MFeatureExtractor(object):
  """Extracts YouTube8M features for RGB frames.
  First time constructing this class will create directory `yt8m` inside your
  home directory, and will download inception model (85 MB) and YouTube8M PCA
  matrix (15 MB). If you want to use another directory, then pass it to argument
  `model_dir` of constructor.
  If the model_dir exist and contains the necessary files, then files will be
  re-used without download.
  Usage Example:
      from PIL import Image
      import numpy
      # Instantiate extractor. Slow if called first time on your machine, as it
      # needs to download 100 MB.
      extractor = YouTube8MFeatureExtractor()
      image_file = os.path.join(extractor._model_dir, 'cropped_panda.jpg')
      im = numpy.array(Image.open(image_file))
      features = extractor.extract_rgb_frame_features(im)
  ** Note: OpenCV reverses the order of channels (i.e. orders channels as BGR
  instead of RGB). If you are using OpenCV, then you must do:
      im = im[:, :, ::-1]  # Reverses order on last (i.e. channel) dimension.
  then call `extractor.extract_rgb_frame_features(im)`
  """

  def __init__(self, model_dir=MODEL_DIR):
    # Create MODEL_DIR if not created.
    self._model_dir = model_dir
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # Load PCA Matrix.
    download_path = self._maybe_download(YT8M_PCA_MAT)
    pca_mean = os.path.join(self._model_dir, 'mean.npy')
    if not os.path.exists(pca_mean):
      tarfile.open(download_path, 'r:gz').extractall(model_dir)
    self._load_pca()

    # Load Inception Network
    download_path = self._maybe_download(INCEPTION_TF_GRAPH)
    inception_proto_file = os.path.join(self._model_dir,
                                        'classify_image_graph_def.pb')
    if not os.path.exists(inception_proto_file):
      tarfile.open(download_path, 'r:gz').extractall(model_dir)
    self._load_inception(inception_proto_file)

  def extract_rgb_frame_features(self, frame_rgb, apply_pca=True):
    """Applies the YouTube8M feature extraction over an RGB frame.
    This passes `frame_rgb` to inception3 model, extracting hidden layer
    activations and passing it to the YouTube8M PCA transformation.
    Args:
      frame_rgb: numpy array of uint8 with shape (height, width, channels) where
        channels must be 3 (RGB), and height and weight can be anything, as the
        inception model will resize.
      apply_pca: If not set, PCA transformation will be skipped.
    Returns:
      Output of inception from `frame_rgb` (2048-D) and optionally passed into
      YouTube8M PCA transformation (1024-D).
    """
    assert len(frame_rgb.shape) == 3
    assert frame_rgb.shape[2] == 3  # 3 channels (R, G, B)
    with self._inception_graph.as_default():
      if apply_pca:
        frame_features = self.session.run(
            'pca_final_feature:0', feed_dict={'DecodeJpeg:0': frame_rgb})
      else:
        frame_features = self.session.run(
            'pool_3/_reshape:0', feed_dict={'DecodeJpeg:0': frame_rgb})
        frame_features = frame_features[0]
    return frame_features

  def apply_pca(self, frame_features):
    """Applies the YouTube8M PCA Transformation over `frame_features`.
    Args:
      frame_features: numpy array of floats, 2048 dimensional vector.
    Returns:
      1024 dimensional vector as a numpy array.
    """
    # Subtract mean
    feats = frame_features - self.pca_mean

    # Multiply by eigenvectors.
    feats = feats.reshape((1, 2048)).dot(self.pca_eigenvecs).reshape((1024,))

    # Whiten
    feats /= numpy.sqrt(self.pca_eigenvals + 1e-4)
    return feats

  def _maybe_download(self, url):
    """Downloads `url` if not in `_model_dir`."""
    filename = os.path.basename(url)
    download_path = os.path.join(self._model_dir, filename)
    if os.path.exists(download_path):
      return download_path

    def _progress(count, block_size, total_size):
      sys.stdout.write(
          '\r>> Downloading %s %.1f%%' %
          (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    urllib.request.urlretrieve(url, download_path, _progress)
    statinfo = os.stat(download_path)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return download_path

  def _load_inception(self, proto_file):
    graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
    self._inception_graph = tf.Graph()
    with self._inception_graph.as_default():
      _ = tf.import_graph_def(graph_def, name='')
      self.session = tf.Session()
      Frame_Features = self.session.graph.get_tensor_by_name(
          'pool_3/_reshape:0')
      Pca_Mean = tf.constant(value=self.pca_mean, dtype=tf.float32)
      Pca_Eigenvecs = tf.constant(value=self.pca_eigenvecs, dtype=tf.float32)
      Pca_Eigenvals = tf.constant(value=self.pca_eigenvals, dtype=tf.float32)
      Feats = Frame_Features[0] - Pca_Mean
      Feats = tf.reshape(
          tf.matmul(tf.reshape(Feats, [1, 2048]), Pca_Eigenvecs), [
              1024,
          ])
      tf.divide(Feats, tf.sqrt(Pca_Eigenvals + 1e-4), name='pca_final_feature')

  def _load_pca(self):
    self.pca_mean = numpy.load(os.path.join(self._model_dir, 'mean.npy'))[:, 0]
    self.pca_eigenvals = numpy.load(
        os.path.join(self._model_dir, 'eigenvals.npy'))[:1024, 0]
    self.pca_eigenvecs = numpy.load(
        os.path.join(self._model_dir, 'eigenvecs.npy')).T[:, :1024]

def transcribe(wavfile):
    r = sr.Recognizer()
    # use wavfile as the audio source (must be .wav file)
    with sr.AudioFile(wavfile) as source:
        #extract audio data from the file
        audio = r.record(source)                    

    transcript=r.recognize_sphinx(audio)
    print(transcript)
    return transcript

# Instantiate extractor. Slow if called first time on your machine, as it
# needs to download 100 MB.
def y8m_featurize(videofile, process_dir, help_dir, fast_model):

  now=os.getcwd()
  # PREPROCESSING
  #############################################
  # metadata (should be .mp4)
  clip = VideoFileClip(videofile)
  duration = clip.duration
  videodata=skvideo.io.vread(videofile)
  frames, rows, cols, channels = videodata.shape
  metadata=skvideo.io.ffprobe(videofile)
  frame=videodata[0]
  r,c,ch=frame.shape

  try:
      os.mkdir('output')
      os.chdir('output')
      outputdir=os.getcwd()
  except:
      shutil.rmtree('output')
      os.mkdir('output')
      os.chdir('output')
      outputdir=os.getcwd()

  #write all the images every 10 frames in the video 
  for i in range(0,len(videodata),25):
      #row, col, channels
      skvideo.io.vwrite("output"+str(i)+".png", videodata[i])
      
  listdir=os.listdir()
  (r,c,ch)=cv2.imread(listdir[0]).shape
  img=numpy.zeros((r,c,ch))
  iterations=0
  #take first image as a background image 
  background=cv2.imread(listdir[1])
  image_features=numpy.zeros(1024)
  image_features2=np.zeros(63)
  image_transcript=''
  for i in range(len(listdir)):
      if listdir[i][-4:]=='.png':
          os.chdir(outputdir)
          frame_new=cv2.imread(listdir[i])
          print(os.getcwd())
          print(listdir[i])
          print(frame)
          img=img+frame_new
          iterations=iterations+1

          # get features 
          extractor = YouTube8MFeatureExtractor(model_dir=help_dir)
          im = numpy.array(Image.open(listdir[i]))
          image_features_temp = extractor.extract_rgb_frame_features(im)
          image_features=image_features+image_features_temp
          ttranscript, tfeatures, tlabels = tff.tesseract_featurize(listdir[i])
          image_transcript=image_transcript+ttranscript 
          image_features2=image_features2+tfeatures 
          #os.remove(listdir[i])

  # averaged image features
  image_features=(1/iterations)*image_features
  image_features2=(1/iterations)*image_features2

  # averaged image over background             
  img=(1/iterations)*img-background
  skvideo.io.vwrite("output.png", img)
  extractor=YouTube8MFeatureExtractor(model_dir=help_dir)
  im = numpy.array(Image.open('output.png'))
  avg_image_features = extractor.extract_rgb_frame_features(im)
  os.remove('output.png')
  os.chdir(now)

  video_features=image_features+avg_image_features
  video_labels=list()
  for i in range(len(video_features)):
      video_labels.append('Y8M_feature_%s'%(str(i+1)))

  avg_image_labels2=list()
  for i in range(len(tlabels)):
      avg_image_labels2.append('avg_imgtranscript_'+tlabels[i])

  # make wavfile from video file and get average AudioSet embedding features 
  wavfile = videofile[0:-4]+'.wav'
  os.system('ffmpeg -i %s %s'%(videofile,wavfile))
  audio_features, audio_labels = af.audioset_featurize(wavfile, audioset_dir, process_dir)

  a_features=numpy.zeros(len(audio_features[0]))
  for i in range(len(audio_features)):
      a_features=a_features+audio_features[i]

  # average all the audioset features
  audio_features=(1/len(audio_features[0]))*a_features
  audio_labels=list()
  for i in range(len(audio_features)):
    audio_labels.append('audioset_feature_%s'%(str(i+1)))

  # extract text and get using FastText model 
  transcript = transcribe(wavfile)
  text_features, text_labels = ff.fast_featurize(transcript, fast_model)

  features=numpy.append(video_features,image_features2)
  features=numpy.append(features, audio_features)
  features=numpy.append(features,text_features)
  labels=video_labels+avg_image_labels2+audio_labels+text_labels 

  return features, labels, transcript, image_transcript




 