## Production directory

From the command line, generate relevant repository for a trained machine learning model.

Note that this assumes a standard audio feature array (for now), but it is expected we can adapt this to the future schema presented in this repository (to accomodate video, text, image, and csv types - and other audio feature embeddings).

## Settings.json

If you set create_YAML == True, the production folder will automatically generate a production-ready repository from the model.py script. This is the easiest way to spin up new models into production.

## How to call from command line 

```python3 
python3 create_yaml.py [sampletype] [simple_model_name] [model_name] [jsonfilename] [class 1] [class 2] ... [class N]
```

Where:
- sampletype = 'audio' | 'video' | 'text' | 'image' | 'csv'
- simple_model name = any string for a common name for model (e.g. gender for male_female_sc_classification.pickle)
- model_name = male_female_sc_classification.pickle
- jsonfile_name = male_female_sc_classification.json (JSON file with model information regarding accuracy, etc.)
- classes = ['male', 'female']

## Quick example

Assuming you have trained a model (stressed_calm_sc_classification.json) in the model directory (audio_models) with stressed_features.json and calm_features.json test cases, you can run the code as follows.

```python3
python3 create_yaml.py audio stress stressed_calm_sc_classification.pickle stressed_calm_sc_classification.json stressed calm  
```

This will then create a repository nlx-model-stress that can be used for production purposes. Note that automated tests 
require testing data, and some of this data can be provided during model training.

Now you can test the docker container by going to the directory and running docker build . 

```
cd nlx-model-stress
docker bulid .
```

if all is good, the tests should indicate it (like below).
```docker
Sending build context to Docker daemon  1.289MB
Step 1/12 : FROM gcr.io/arctic-robot-192514/nlx-base
 ---> 43fdd5d2f70e
Step 2/12 : WORKDIR /usr/src/app
 ---> Using cache
 ---> 85479d881540
Step 3/12 : ADD . /usr/src/app
 ---> bcad88174079
Step 4/12 : COPY requirements.txt ./
 ---> b005dd8f1269
Step 5/12 : RUN pip install --no-cache-dir -r requirements.txt
 ---> Running in 86a3db5ab3dc
Collecting kafka-python (from -r requirements.txt (line 1))
  Downloading https://files.pythonhosted.org/packages/82/39/aebe3ad518513bbb2260dd84ac21e5c30af860cc4c95b32acbd64b9d9d0d/kafka_python-1.4.6-py2.py3-none-any.whl (259kB)
Collecting numpy (from -r requirements.txt (line 2))
  Downloading https://files.pythonhosted.org/packages/87/2d/e4656149cbadd3a8a0369fcd1a9c7d61cc7b87b3903b85389c70c989a696/numpy-1.16.4-cp36-cp36m-manylinux1_x86_64.whl (17.3MB)
Collecting matplotlib (from -r requirements.txt (line 3))
  Downloading https://files.pythonhosted.org/packages/da/83/d989ee20c78117c737ab40e0318ea221f1aed4e3f5a40b4f93541b369b93/matplotlib-3.1.0-cp36-cp36m-manylinux1_x86_64.whl (13.1MB)
Collecting scipy (from -r requirements.txt (line 4))
  Downloading https://files.pythonhosted.org/packages/72/4c/5f81e7264b0a7a8bd570810f48cd346ba36faedbd2ba255c873ad556de76/scipy-1.3.0-cp36-cp36m-manylinux1_x86_64.whl (25.2MB)
Collecting sklearn (from -r requirements.txt (line 5))
  Downloading https://files.pythonhosted.org/packages/1e/7a/dbb3be0ce9bd5c8b7e3d87328e79063f8b263b2b1bfa4774cb1147bfcd3f/sklearn-0.0.tar.gz
Collecting hmmlearn (from -r requirements.txt (line 6))
  Downloading https://files.pythonhosted.org/packages/d7/c5/91b43156b193d180ed94069269bcf88d3c7c6e54514a8482050fa9995e10/hmmlearn-0.2.2.tar.gz (146kB)
Collecting simplejson (from -r requirements.txt (line 7))
  Downloading https://files.pythonhosted.org/packages/e3/24/c35fb1c1c315fc0fffe61ea00d3f88e85469004713dab488dee4f35b0aff/simplejson-3.16.0.tar.gz (81kB)
Collecting eyed3 (from -r requirements.txt (line 8))
  Downloading https://files.pythonhosted.org/packages/75/66/b1d97f06499199b3b3dfcd25e3f000c4256e7f313b3a0aae2e3d24e6acb7/eyeD3-0.8.10-py2.py3-none-any.whl (147kB)
Collecting pydub (from -r requirements.txt (line 9))
  Downloading https://files.pythonhosted.org/packages/79/db/eaf620b73a1eec3c8c6f8f5b0b236a50f9da88ad57802154b7ba7664d0b8/pydub-0.23.1-py2.py3-none-any.whl
Collecting nltk (from -r requirements.txt (line 10))
  Downloading https://files.pythonhosted.org/packages/8d/5d/825889810b85c303c8559a3fd74d451d80cf3585a851f2103e69576bf583/nltk-3.4.3.zip (1.4MB)
Collecting colorama (from -r requirements.txt (line 11))
  Downloading https://files.pythonhosted.org/packages/4f/a6/728666f39bfff1719fc94c481890b2106837da9318031f71a8424b662e12/colorama-0.4.1-py2.py3-none-any.whl
Collecting libmagic (from -r requirements.txt (line 12))
  Downloading https://files.pythonhosted.org/packages/83/86/419ddfc3879b4565a60e0c75b6d19baec48428cbc2f15aca5320b3d136f6/libmagic-1.0.tar.gz
Collecting librosa (from -r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/e9/7e/7a0f66f79a70a0a4c163ecf30429f6c1644c88654f135a9eee0bda457626/librosa-0.6.3.tar.gz (1.6MB)
Collecting pymongo (from -r requirements.txt (line 14))
  Downloading https://files.pythonhosted.org/packages/fb/4a/586826433281ca285f0201235fccf63cc29a30fa78bcd72b6a34e365972d/pymongo-3.8.0-cp36-cp36m-manylinux1_x86_64.whl (416kB)
Collecting dnspython (from -r requirements.txt (line 15))
  Downloading https://files.pythonhosted.org/packages/ec/d3/3aa0e7213ef72b8585747aa0e271a9523e713813b9a20177ebe1e939deb0/dnspython-1.16.0-py2.py3-none-any.whl (188kB)
Collecting requests (from -r requirements.txt (line 16))
  Downloading https://files.pythonhosted.org/packages/51/bd/23c926cd341ea6b7dd0b2a00aba99ae0f828be89d72b2190f27c11d4b7fb/requests-2.22.0-py2.py3-none-any.whl (57kB)
Collecting pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 (from matplotlib->-r requirements.txt (line 3))
  Downloading https://files.pythonhosted.org/packages/dd/d9/3ec19e966301a6e25769976999bd7bbe552016f0d32b577dc9d63d2e0c49/pyparsing-2.4.0-py2.py3-none-any.whl (62kB)
Collecting kiwisolver>=1.0.1 (from matplotlib->-r requirements.txt (line 3))
  Downloading https://files.pythonhosted.org/packages/f8/a1/5742b56282449b1c0968197f63eae486eca2c35dcd334bab75ad524e0de1/kiwisolver-1.1.0-cp36-cp36m-manylinux1_x86_64.whl (90kB)
Collecting python-dateutil>=2.1 (from matplotlib->-r requirements.txt (line 3))
  Downloading https://files.pythonhosted.org/packages/41/17/c62faccbfbd163c7f57f3844689e3a78bae1f403648a6afb1d0866d87fbb/python_dateutil-2.8.0-py2.py3-none-any.whl (226kB)
Collecting cycler>=0.10 (from matplotlib->-r requirements.txt (line 3))
  Downloading https://files.pythonhosted.org/packages/f7/d2/e07d3ebb2bd7af696440ce7e754c59dd546ffe1bbe732c8ab68b9c834e61/cycler-0.10.0-py2.py3-none-any.whl
Collecting scikit-learn (from sklearn->-r requirements.txt (line 5))
  Downloading https://files.pythonhosted.org/packages/85/04/49633f490f726da6e454fddc8e938bbb5bfed2001681118d3814c219b723/scikit_learn-0.21.2-cp36-cp36m-manylinux1_x86_64.whl (6.7MB)
Collecting six (from eyed3->-r requirements.txt (line 8))
  Downloading https://files.pythonhosted.org/packages/73/fb/00a976f728d0d1fecfe898238ce23f502a721c0ac0ecfedb80e0d88c64e9/six-1.12.0-py2.py3-none-any.whl
Collecting python-magic (from eyed3->-r requirements.txt (line 8))
  Downloading https://files.pythonhosted.org/packages/42/a1/76d30c79992e3750dac6790ce16f056f870d368ba142f83f75f694d93001/python_magic-0.4.15-py2.py3-none-any.whl
Collecting audioread>=2.0.0 (from librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/2e/0b/940ea7861e0e9049f09dcfd72a90c9ae55f697c17c299a323f0148f913d2/audioread-2.1.8.tar.gz
Collecting joblib>=0.12 (from librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/cd/c1/50a758e8247561e58cb87305b1e90b171b8c767b15b12a1734001f41d356/joblib-0.13.2-py2.py3-none-any.whl (278kB)
Collecting decorator>=3.0.0 (from librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/5f/88/0075e461560a1e750a0dcbf77f1d9de775028c37a19a346a6c565a257399/decorator-4.4.0-py2.py3-none-any.whl
Collecting resampy>=0.2.0 (from librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/14/b6/66a06d85474190b50aee1a6c09cdc95bb405ac47338b27e9b21409da1760/resampy-0.2.1.tar.gz (322kB)
Collecting numba>=0.38.0 (from librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/14/8b/eff047afc373ff9f72a6ab29952a51ee076b3640fcda5a6d7500899cb9d4/numba-0.44.0-cp36-cp36m-manylinux1_x86_64.whl (3.4MB)
Collecting certifi>=2017.4.17 (from requests->-r requirements.txt (line 16))
  Downloading https://files.pythonhosted.org/packages/69/1b/b853c7a9d4f6a6d00749e94eb6f3a041e342a885b87340b79c1ef73e3a78/certifi-2019.6.16-py2.py3-none-any.whl (157kB)
Collecting chardet<3.1.0,>=3.0.2 (from requests->-r requirements.txt (line 16))
  Downloading https://files.pythonhosted.org/packages/bc/a9/01ffebfb562e4274b6487b4bb1ddec7ca55ec7510b22e4c51f14098443b8/chardet-3.0.4-py2.py3-none-any.whl (133kB)
Collecting idna<2.9,>=2.5 (from requests->-r requirements.txt (line 16))
  Downloading https://files.pythonhosted.org/packages/14/2c/cd551d81dbe15200be1cf41cd03869a46fe7226e7450af7a6545bfc474c9/idna-2.8-py2.py3-none-any.whl (58kB)
Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 (from requests->-r requirements.txt (line 16))
  Downloading https://files.pythonhosted.org/packages/e6/60/247f23a7121ae632d62811ba7f273d0e58972d75e58a94d329d51550a47d/urllib3-1.25.3-py2.py3-none-any.whl (150kB)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/site-packages (from kiwisolver>=1.0.1->matplotlib->-r requirements.txt (line 3)) (39.2.0)
Collecting llvmlite>=0.29.0 (from numba>=0.38.0->librosa->-r requirements.txt (line 13))
  Downloading https://files.pythonhosted.org/packages/09/f1/4fa99c2079132da8694361fb9a19094616f1ba2c2eae610379e75394575f/llvmlite-0.29.0-cp36-cp36m-manylinux1_x86_64.whl (20.4MB)
Installing collected packages: kafka-python, numpy, pyparsing, kiwisolver, six, python-dateutil, cycler, matplotlib, scipy, joblib, scikit-learn, sklearn, hmmlearn, simplejson, python-magic, eyed3, pydub, nltk, colorama, libmagic, audioread, decorator, llvmlite, numba, resampy, librosa, pymongo, dnspython, certifi, chardet, idna, urllib3, requests
  Running setup.py install for sklearn: started
    Running setup.py install for sklearn: finished with status 'done'
  Running setup.py install for hmmlearn: started
    Running setup.py install for hmmlearn: finished with status 'done'
  Running setup.py install for simplejson: started
    Running setup.py install for simplejson: finished with status 'done'
  Running setup.py install for nltk: started
    Running setup.py install for nltk: finished with status 'done'
  Running setup.py install for libmagic: started
    Running setup.py install for libmagic: finished with status 'done'
  Running setup.py install for audioread: started
    Running setup.py install for audioread: finished with status 'done'
  Running setup.py install for resampy: started
    Running setup.py install for resampy: finished with status 'done'
  Running setup.py install for librosa: started
    Running setup.py install for librosa: finished with status 'done'
Successfully installed audioread-2.1.8 certifi-2019.6.16 chardet-3.0.4 colorama-0.4.1 cycler-0.10.0 decorator-4.4.0 dnspython-1.16.0 eyed3-0.8.10 hmmlearn-0.2.2 idna-2.8 joblib-0.13.2 kafka-python-1.4.6 kiwisolver-1.1.0 libmagic-1.0 librosa-0.6.3 llvmlite-0.29.0 matplotlib-3.1.0 nltk-3.4.3 numba-0.44.0 numpy-1.16.4 pydub-0.23.1 pymongo-3.8.0 pyparsing-2.4.0 python-dateutil-2.8.0 python-magic-0.4.15 requests-2.22.0 resampy-0.2.1 scikit-learn-0.21.2 scipy-1.3.0 simplejson-3.16.0 six-1.12.0 sklearn-0.0 urllib3-1.25.3
You are using pip version 10.0.1, however version 19.1.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
Removing intermediate container 86a3db5ab3dc
 ---> a111cc8442ab
Step 6/12 : RUN python test.py
 ---> Running in 692190381cd6
/usr/local/lib/python3.6/site-packages/sklearn/base.py:306: UserWarning: Trying to unpickle estimator LogisticRegression from version 0.19.1 when using version 0.21.2. This might lead to breaking code or invalid results. Use at your own risk.
  UserWarning)
test.py:18: ResourceWarning: unclosed file <_io.TextIOWrapper name='/usr/src/app/test/calm_features.json' mode='r' encoding='UTF-8'>
  features_data = open(test_dir + '/calm_features.json', 'r').read()
.test.py:12: ResourceWarning: unclosed file <_io.TextIOWrapper name='/usr/src/app/test/stressed_features.json' mode='r' encoding='UTF-8'>
  features_data = open(test_dir + '/stressed_features.json', 'r').read()
.
----------------------------------------------------------------------
Ran 2 tests in 0.008s

OK
Removing intermediate container 692190381cd6
 ---> d9d77db7c76a
Step 7/12 : ENV KAFKA_HOST="kafka-service:9092"
 ---> Running in bbd6c858fd4b
Removing intermediate container bbd6c858fd4b
 ---> a16daf15b937
Step 8/12 : ENV KAFKA_INCOMING_TOPIC="REQUESTED_MODEL_STRESS"
 ---> Running in c6dddc6f9150
Removing intermediate container c6dddc6f9150
 ---> 2d934799e7a5
Step 9/12 : ENV KAFKA_OUTGOING_TOPIC="CREATED_MODEL_STRESS"
 ---> Running in aff19ad529cd
Removing intermediate container aff19ad529cd
 ---> f3cb39339f58
Step 10/12 : ENV MONGO_URL="mongo-service:27017"
 ---> Running in 0b7909e62872
Removing intermediate container 0b7909e62872
 ---> b84357217968
Step 11/12 : ENV MONGO_DB="nlx-data"
 ---> Running in ac99ce6e4b77
Removing intermediate container ac99ce6e4b77
 ---> b745766e535b
Step 12/12 : CMD ["python", "-u", "/usr/src/app/server.py"]
 ---> Running in 83f81c988427
Removing intermediate container 83f81c988427
 ---> 0137f8d69282
Successfully built 0137f8d69282

```

## Fixing test failures 
If you have a fail in testing, it's likely due to outputs in the model not being correct. In this case, you may need to manually tune the outputs.
```python3
======================================================================
FAIL: test_calm_classification (__main__.ClassifyTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test.py", line 21, in test_calm_classification
    self.assertEqual(results, 1)
AssertionError: 'calm' != 1

======================================================================
FAIL: test_stressed_classification (__main__.ClassifyTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test.py", line 15, in test_stressed_classification
    self.assertEqual(results, 0)
AssertionError: 'stressed' != 0

----------------------------------------------------------------------
Ran 2 tests in 0.011s

FAILED (failures=2)
```

This can be corrected by changing this code (test.py in nlx-model-stress repo):
```python3
import os
import unittest
import classify
import json
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(cwd + '/test')

class ClassifyTest(unittest.TestCase):
  def test_stressed_classification(self):
    features_data = open(test_dir + '/stressed_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 0)

  def test_calm_classification(self):
    features_data = open(test_dir + '/calm_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 1)

if __name__ == '__main__':
  unittest.main()
```


to this code:

```python3
import os
import unittest
import classify
import json
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(cwd + '/test')

class ClassifyTest(unittest.TestCase):
  def test_stressed_classification(self):
    features_data = open(test_dir + '/stressed_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 'stressed')

  def test_calm_classification(self):
    features_data = open(test_dir + '/calm_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 'calm')

if __name__ == '__main__':
  unittest.main()
```
 
## Work in progress
* add in all types of feature arrays and data types (to production training) - rename to include type of file (e.g. nlx-model-audio-age vs. nlx-model-video-age).
* adapt testing scripts in this folder to keras/devol and ludwig (to allow for any model type to be spun up into production).
* make 100 different types of gifs to make the automated documentation more interesting!! :-)
