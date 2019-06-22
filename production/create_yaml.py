'''
From the command line, generate relevant repository for a trained machine 
learning model.
This assumes a standard feature array (for now), but it is expected we can 
adapt this to the future schema presented in this repository.
######################################################
           HOW TO CALL FROM COMMAND LINE 
######################################################
python3 create_yaml.py [sampletype] [simple_model_name] [model_name] [jsonfilename] [class 1] [class 2] ... [class N]
Where:
	sampletype = 'audio' | 'video' | 'text' | 'image' | 'csv'
	simple_model name = any string for a common name for model (e.g. gender for male_female_sc_classification.pickle)
	model_name = male_female_sc_classification.pickle
	jsonfile_name = male_female_sc_classification.json (JSON file with model information regarding accuracy, etc.)
	classes = ['male', 'female']
Example:
	python3 create_yaml.py audio stress stressed_calm_sc_classification.pickle stressed_calm_sc_classification.json stressed calm  
This will then create a repository nlx-model-stress that can be used for production purposes. Note that automated tests 
require testing data, and some of this data can be provided during model training.
 '''

import os, sys, shutil, json

def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

def create_classifyfile(model_name):
	# this should be the same for every repository if the files are named classify.py
	g=open('classify.py', 'w')
	g.write('import os \n')
	g.write('import pickle \n')
	g.write('import json \n')
	g.write('import numpy as np \n')
	g.write('\n')
	g.write('# configure the models\n')
	g.write('cwd = os.path.dirname(os.path.abspath(__file__)) \n')
	g.write("data_dir = os.path.abspath(cwd + '/data') \n")
	g.write("model_data = open(data_dir + '/%s', 'rb') \n"%(model_name))
	g.write("model = pickle.load(model_data) \n")
	g.write("model_data.close()\n\n")
	g.write('# classify \n')
	g.write('def classify(features):\n')
	g.write('  results = model.predict(features)\n')
	g.write('  return results[0]')
	g.close()

def create_YAML(reponame):
	# this should be the same except for file name 
	# filename=nlx-model-stress
	g=open('cloudbuild.yaml', 'w')
	g.write('steps: \n')
	g.write('- name: debian\n')
	g.write("  args: ['sed', '-i', 's#https://github.com/NeuroLexDiagnostics/pyAudioLex#https://source.developers.google.com/p/arctic-robot-192514/r/github-neurolexdiagnostics-pyaudiolex#g', '.gitmodules'] \n")
	g.write("- name: 'gcr.io/cloud-builders/git'\n")
	g.write("  args: ['submodule', 'update', '--init', '--recursive']\n")
	g.write("- name: 'ubuntu'\n")
	g.write("  args: ['ls', '-lsa', 'pyAudioLex']\n")
	g.write("- name: 'gcr.io/cloud-builders/docker'\n")
	g.write("  args: ['build', '-t', 'gcr.io/$PROJECT_ID/%s:$COMMIT_SHA', '.']\n"%(reponame))
	g.write("images: ['gcr.io/$PROJECT_ID/%s:$COMMIT_SHA']\n"%(reponame))
	g.close()
	
def create_dockercompose():
	g=open('docker-compose.yml', 'w')
	g.write("version: '3'\n")
	g.write("services:\n")
	g.write("  zookeeper-service:\n")
	g.write("    image: confluentinc/cp-zookeeper:4.0.0\n")
	g.write("    ports:\n")
	g.write('      - "2181:2181"\n')
	g.write("    environment:\n")
	g.write("      ZOOKEEPER_CLIENT_PORT: 2181\n")
	g.write("      ZOOKEEPER_TICK_TIME: 2000\n")
	g.write("  kafka-service:\n")
	g.write("    image: confluentinc/cp-kafka:4.0.0\n")
	g.write("    ports:\n")
	g.write('      - "29092:29092"\n')
	g.write("    depends_on:\n")
	g.write("      - zookeeper-service\n")
	g.write("    environment:\n")
	g.write("      KAFKA_BROKER_ID: 1\n")
	g.write("      KAFKA_ZOOKEEPER_CONNECT: zookeeper-service:2181\n")
	g.write("      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-service:9092\n")
	g.write("      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1\n")
	g.write("  mongo-service:\n")
	g.write("    image: launcher.gcr.io/google/mongodb3\n")
	g.write("    ports:\n")
	g.write('      - "27017:27017"\n')
	g.close()

def create_docker(incoming_topic, outgoing_topic):
	g=open('Dockerfile','w')
	g.write("FROM gcr.io/arctic-robot-192514/nlx-base\n")
	g.write("WORKDIR /usr/src/app\n")
	g.write("ADD . /usr/src/app\n")
	g.write("# install dependencies\n")
	g.write("COPY requirements.txt ./\n")
	g.write("RUN pip install --no-cache-dir -r requirements.txt\n")
	g.write("# run tests\n")
	g.write("RUN python test.py\n")
	g.write("# env\n")
	g.write('ENV KAFKA_HOST="kafka-service:9092"\n')
	g.write('ENV KAFKA_INCOMING_TOPIC="%s"\n'%(incoming_topic))
	g.write('ENV KAFKA_OUTGOING_TOPIC="%s"\n'%(outgoing_topic))
	g.write('ENV MONGO_URL="mongo-service:27017"\n')
	g.write('ENV MONGO_DB="nlx-data"\n')
	g.write('CMD ["python", "-u", "/usr/src/app/server.py"]')
	g.close()

def create_makefile(reponame, incoming_topic, outgoing_topic):
	g=open('Makefile','w')
	g.write('build:\n')
	g.write("	docker build . -t %s\n"%(reponame))
	g.write("run:\n")
	g.write("	docker run -d \\ \n")
	g.write("		--network=%s_default \\ \n"%(reponame))
	g.write("		--name=%s_worker_1 \\ \n"%(reponame))
	g.write('		-e "KAFKA_HOST=kafka-service:9092" \\ \n')
	g.write('		-e "KAFKA_INCOMING_TOPIC=%s" \\ \n'%(incoming_topic))
	g.write('		-e "KAFKA_OUTGOING_TOPIC=%s" \\ \n'%(outgoing_topic))
	g.write('		-e "MONGO_URL=mongo-service:27017" \\ \n')
	g.write('		-e "MONGO_DB=nlx-data" \\ \n')
	g.write('		%s\n'%(reponame))
	g.write('stop:\n')
	g.write('	docker stop %s_worker_1 || true\n'%(reponame))
	g.write('	docker-compose stop\n')
	g.write('kill: stop\n')
	g.write('	docker rm %s_worker_1\n'%(reponame))
	g.write('	docker-compose down\n')
	g.write('up:\n')
	g.write('	docker-compose up -d\n')
	g.write("	printf 'Waiting for kafka...\\n' \n")
	g.write("	sleep 15\n")
	g.write("	make setup\n")
	g.write("	make build\n")
	g.write("	make run\n")
	g.write("down:\n")
	g.write("	make kill\n")
	g.write("setup:\n")
	g.write("	docker exec -it %s_kafka-service_1 \\ \n"%(reponame))
	g.write("		/usr/bin/kafka-topics \\ \n")
	g.write("		--create \\ \n")
	g.write("		--zookeeper zookeeper-service:2181 \\ \n")
	g.write("		--replication-factor 1 \\ \n")
	g.write("		--partitions 1 \\ \n")
	g.write("		--topic %s\n"%(incoming_topic))
	g.write("	docker exec -it %s_kafka-service_1 \\ \n"%(reponame))
	g.write("		/usr/bin/kafka-topics \\ \n")
	g.write("		--create \\ \n")
	g.write("		--zookeeper zookeeper-service:2181 \\ \n")
	g.write("		--replication-factor 1 \\ \n")
	g.write("		--partitions 1 \\ \n")
	g.write("		--topic %s \n"%(outgoing_topic))
	g.write("# populate:\n")
	g.write("	# docker exec -it %s_kafka-service_1 /usr/bin/kafka-console-producer --broker-list kafka-service:9092 --topic %s\n"%(reponame, incoming_topic))
	g.write('	# > {"id":""}\n')
	g.close()

def create_processpy(modelname):
	g=open('process.py','w')
	g.write("import os\n")
	g.write("from pymongo import MongoClient\n")
	g.write("import numpy as np\n")
	g.write("import classify\n\n")
	g.write("# configure mongo\n")
	g.write("MONGO_URL = os.environ['MONGO_URL']\n")
	g.write("MONGO_DB = os.environ['MONGO_DB']\n")
	g.write("client = MongoClient(MONGO_URL)\n")
	g.write("db = client[MONGO_DB]\n")
	g.write("v1Features = db.v1Features\n")
	g.write("samples = db.samples\n\n")
	g.write("# process the payload\n")
	g.write("def process(payload):\n")
	g.write("  sampleID = payload['id']\n")
	g.write("  print('Classifying for ' + sampleID)\n\n")
	g.write("  # get features and classify\n")
	g.write('  v1Feature = v1Features.find_one({ "sampleID": sampleID })\n')
	g.write("  # check that we have the feature array\n")
	g.write("  if 'standard_feature_array' in v1Feature['features']['audio']:\n")
	g.write("    features_data = v1Feature['features']['audio']['standard_feature_array']\n")
	g.write("    features = np.array(features_data).reshape(1, -1)\n")
	g.write("    result = classify.classify(features)\n")
	g.write("    # save to mongo\n")
	g.write("    print(f'Inserted into mongo with %s: {result}')\n"%(modelname))
	g.write('    v1Features.find_one_and_update({ "sampleID": sampleID },\n')
	g.write('                                   { "$set": { "model.%s": result } })\n'%(modelname))
	g.write('    samples.find_one_and_update({ "sampleId": sampleID },\n')
	g.write('                                       { "$set": { "model.%s": result } })\n'%(modelname))
	g.write('  # otherwise skip\n')
	g.write("  else:\n")
	g.write("    print('Skipping as could not find the proper features')\n")
	g.close()

def create_readme(reponame, classes, modelname, sampletype, giflink, default_features, labels, modelinfo):
	g=open('readme.md', 'w')
	g.write('# %s \n'%(reponame))
	g.write('This is a repository for modeling %s (%s files). \n\n'%(modelname, sampletype))
	g.write('![](%s)\n\n'%(giflink))
	g.write('This documentation is automatically generated with the create_yaml.py script. :-) \n')
	g.write('## Model performance \n')
	g.write('```')
	g.write(modelinfo)
	g.write('```\n')
	g.write('## Class Predictions \n')
	g.write('```Classes: %s```\n'%(str(classes)))
	g.write('## Feature array \n')
	g.write('```Feature array: %s (%s features)``` \n'%(default_features, str(len(labels))))
	g.write('## Feature labels \n') 
	g.write('```Feature labels: %s ``` \n'%(labels))
	g.close()

def create_requirements():
	g=open('requirements.txt','w')
	g.write('kafka-python\n')
	g.write('numpy\n')
	g.write('matplotlib\n')
	g.write('scipy\n')
	g.write('sklearn\n')
	g.write('hmmlearn\n')
	g.write('simplejson\n')
	g.write('eyed3\n')
	g.write('pydub\n')
	g.write('nltk\n')
	g.write('colorama\n')
	g.write('libmagic\n')
	g.write('librosa\n')
	g.write('pymongo\n')
	g.write('dnspython\n')
	g.write('requests\n')
	g.close()

def create_server(modelname):
	g=open('server.py','w')
	g.write('import os\n')
	g.write('import json\n')
	g.write('from kafka import KafkaConsumer, KafkaProducer\n')
	g.write('import process\n')
	g.write('import requests\n\n')
	g.write('# configure kafka\n')
	g.write("KAFKA_HOST = os.environ['KAFKA_HOST']\n")
	g.write("KAFKA_INCOMING_TOPIC = os.environ['KAFKA_INCOMING_TOPIC']\n")
	g.write("KAFKA_OUTGOING_TOPIC = os.environ['KAFKA_OUTGOING_TOPIC']\n")
	g.write("OBJECT_ENDPOINT = os.environ['OBJECT_ENDPOINT']\n")
	g.write("bootstrap_servers = [KAFKA_HOST]\n")
	g.write("incoming_topic = KAFKA_INCOMING_TOPIC\n")
	g.write("outgoing_topic = KAFKA_OUTGOING_TOPIC\n")
	g.write("producer = KafkaProducer(bootstrap_servers=bootstrap_servers)\n")
	g.write("consumer = KafkaConsumer(incoming_topic, bootstrap_servers=bootstrap_servers)\n")
	g.write("print(f'Microservice listening to topic: {incoming_topic}...')\n")
	g.write("# iterate on the samples\n")
	g.write('# ex: {"id": "NLX-1520555910458794813-1520555959992", "type": "v1Sample"}\n')
	g.write('for msg in consumer:\n')
	g.write('  # output message\n')
	g.write("  print(f'New message: {msg.value}')\n")
	g.write("  # read the payload and process\n")
	g.write("  payload = json.loads(msg.value)\n")
	g.write("  # fix issues with poorly formed payloads\n")
	g.write("  if 'type' not in payload:\n")
	g.write("    payload['type'] = 'sample'\n")
	g.write("  if 'id' not in payload:\n")
	g.write("    payload['id'] = payload['sampleId']\n")
	g.write("  sampleId = payload['id']\n")
	g.write("  print(payload)\n")
	g.write("  # process the payload\n")
	g.write("  process.process(payload)\n")
	g.write("  # asynchronously send the status\n")
	g.write("  try:\n")
	g.write("    print('Submitting status for sample: ' + sampleId)\n")
	g.write("    r = requests.post(OBJECT_ENDPOINT + '/status/samples/' + sampleId + '/modeled-%s')\n"%(modelname))
	g.write("  except:\n")
	g.write("    print('Failed to submit status for sample: ' + sampleId)\n")
	g.write("  # send sample to the next topic\n")
	g.write("  print(f'Submitting message on {outgoing_topic}')\n")
	g.write("  producer.send(outgoing_topic, msg.value)\n")
	g.close()

def create_test(classes):
	g=open('test.py', 'w')
	g.write('import os\n')
	g.write('import unittest\n')
	g.write('import classify\n')
	g.write('import json\n')
	g.write('import numpy as np\n\n')
	g.write('cwd = os.path.dirname(os.path.abspath(__file__))\n')
	g.write("test_dir = os.path.abspath(cwd + '/test')\n\n")
	g.write("class ClassifyTest(unittest.TestCase):\n")
	for i in range(len(classes)):
		g.write("  def test_%s_classification(self):\n"%(classes[i]))
		g.write("    features_data = open(test_dir + '/%s_features.json', 'r').read()\n"%(classes[i]))
		g.write("    features = np.array(json.loads(features_data)).reshape(1, -1)\n")
		g.write("    results = classify.classify(features)\n")
		g.write("    self.assertEqual(results, %s)\n\n"%(str(i)))
	g.write("if __name__ == '__main__':\n")
	g.write("  unittest.main()\n")

def create_init1():
	g=open('__init__.py','w')
	g.write('import pyAudioLex')
	g.close()

### THESE ARE TO GO IN TEST FOLDER 

def create_init2():
	g=open('__init__.py','w')
	g.write('from .. import pyAudioLex')
	g.close()

def featurize_training(classes):
	g=open('featurize_training_data.py','w')
	g.write('import os\n')
	g.write('import json\n')
	g.write('from .pyAudioLex import process_audio\n')
	g.write('cwd = os.path.dirname(os.path.abspath(__file__))\n')
	for i in range(len(classes)):
		g.write("%s_training_folder = os.path.join(cwd, '..', 'data/training/%s \n')"%(classes[i], classes[i]))
	g.write('# process file\n')
	g.write('def process_file(filepath):\n')
	g.write("  print('Processing... ' + filepath)\n")
	g.write("  results = process_audio(filepath)\n")
	g.write("  print(json.dumps(results['standard_feature_array'].tolist()))\n")
	g.write("  return results\n\n")
	g.write("# process all the data in a folder\n")
	g.write("def process_dataset(folder):\n")
	g.write("  files = os.listdir(folder)\n")
	g.write("  for file in files:\n")
	g.write("    filepath = folder + '/' + file\n")
	g.write("    process_file(filepath)\n")
	g.write("    break\n")
	g.write("process_file(os.path.join(%s_training_folder, '%s.wav')) \n"%(classes[0], classes[0]))
	g.close()

def populate_mongo(classes):
	g=open('populate_mongo.py','w')
	g.write('import os\n')
	g.write('from pymongo import MongoClient\n')
	g.write('import featurize_training_data\n\n')
	g.write('cwd = os.path.dirname(os.path.abspath(__file__))\n')
	for i in range(len(classes)):
		g.write("%s_training_folder = os.path.join(cwd, '..', 'data/training/%s')\n"%(classes[i], classes[i]))
	g.write("# configure mongo\n")
	g.write("MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://mongo-service:27017')\n")
	g.write("MONGO_DB = os.environ.get('MONGO_DB', 'nlx-data')\n")
	g.write("client = MongoClient(MONGO_URL)\n")
	g.write("db = client[MONGO_DB]\n")
	g.write("v1Features = db.v1Features\n")
	g.write("results = featurize_training_data.process_file(os.path.join(%s_training_folder, '%s.wav'))\n"%(classes[0], classes[0]))
	g.write("print(results)")
	g.close()

###############################################################
## 	             CREATE DIRECTORIES.  		     ## 
###############################################################

# sampletype = 'audio' 
# feature_array = 'standard_audio_features' [left out here, assumed]
# common_model_name = 'stress' 
# repo_name = nlx-model-stress
# model_name = 'stressed_calm_sc_classification.pickle'
# jsonfile_name = 'stressed_calm_sc_classification.pickle'
# model_dir = /Users/jimschwoebel/desktop/voice_modeling/audio_models/
# classes = ['stressed', 'calm']

sampletype = sys.argv[1]
common_model_name= sys.argv[2]
repo_name = 'nlx-model-'+sampletype+'-'+common_model_name 
model_name = sys.argv[3]
jsonfile_name = sys.argv[4]
count=5
classes=list()
while True:
	try:
		classes.append(sys.argv[count])
	except:
		break
	count=count+1

cur_dir=os.getcwd()
model_dir=prev_dir(cur_dir)+'/models/%s_models/'%(sampletype)

# initialize classes and variables 
temp=model_name.split('_')

incoming_topic='REQUESTED_MODEL_%s'%(common_model_name.upper())
outgoing_topic='CREATED_MODEL_%s'%(common_model_name.upper())
giflink='https://media.giphy.com/media/l3V0x6kdXUW9M4ONq/giphy.gif'

###############################################################
## 	      CREATE FOLDERS / MAKE REPO		     ## 
###############################################################

# make the repo name in production folder
try:
	os.mkdir(repo_name)
	os.chdir(repo_name)
except:
	shutil.rmtree(repo_name)
	os.mkdir(repo_name)
	os.chdir(repo_name)

os.mkdir('data')
# now copy pickle file and .JSON file into directory 
shutil.copy(model_dir+model_name, os.getcwd()+'/data/'+model_name)
shutil.copy(model_dir+jsonfile_name, os.getcwd()+'/data/'+jsonfile_name)
modelinfo=str(json.load(open(os.getcwd()+'/data/'+jsonfile_name)))

os.mkdir('test')
os.chdir('test')
# transfer test cases 
for i in range(len(classes)):
	shutil.copy(model_dir+classes[i]+'_features.json', os.getcwd()+'/'+classes[i]+'_features.json')
shutil.copy(cur_dir+'/helpers/mongo_row.json', os.getcwd()+'/mongo_row.json')
create_init2()
featurize_training(classes)
populate_mongo(classes)
default_features = 'standard features'
standard_labels=['mfcc_1_mean_20ms','mfcc_1_std_20ms', 'mfcc_1_min_20ms', 'mfcc_1_max_20ms',
		        'mfcc_2_mean_20ms','mfcc_2_std_20ms', 'mfcc_2_min_20ms', 'mfcc_2_max_20ms',
		        'mfcc_3_mean_20ms','mfcc_3_std_20ms', 'mfcc_3_min_20ms', 'mfcc_3_max_20ms',
		        'mfcc_4_mean_20ms','mfcc_4_std_20ms', 'mfcc_4_min_20ms', 'mfcc_4_max_20ms',
		        'mfcc_5_mean_20ms','mfcc_5_std_20ms', 'mfcc_5_min_20ms', 'mfcc_5_max_20ms',
		        'mfcc_6_mean_20ms','mfcc_6_std_20ms', 'mfcc_6_min_20ms', 'mfcc_6_max_20ms',
		        'mfcc_7_mean_20ms','mfcc_7_std_20ms', 'mfcc_7_min_20ms', 'mfcc_7_max_20ms',
		        'mfcc_8_mean_20ms','mfcc_8_std_20ms', 'mfcc_8_min_20ms', 'mfcc_8_max_20ms',
		        'mfcc_9_mean_20ms','mfcc_9_std_20ms', 'mfcc_9_min_20ms', 'mfcc_9_max_20ms',
		        'mfcc_10_mean_20ms','mfcc_10_std_20ms', 'mfcc_10_min_20ms', 'mfcc_10_max_20ms',
		        'mfcc_11_mean_20ms','mfcc_11_std_20ms', 'mfcc_11_min_20ms', 'mfcc_11_max_20ms',
		        'mfcc_12_mean_20ms','mfcc_12_std_20ms', 'mfcc_12_min_20ms', 'mfcc_12_max_20ms',
		        'mfcc_13_mean_20ms','mfcc_13_std_20ms', 'mfcc_13_min_20ms', 'mfcc_13_max_20ms',
		        'mfcc_1_delta_mean_20ms','mfcc_1_delta_std_20ms', 'mfcc_1_delta_min_20ms', 'mfcc_1_delta_max_20ms',
		        'mfcc_2_delta_mean_20ms','mfcc_2_delta_std_20ms', 'mfcc_2_delta_min_20ms', 'mfcc_2_delta_max_20ms',
		        'mfcc_3_delta_mean_20ms','mfcc_3_delta_std_20ms', 'mfcc_3_delta_min_20ms', 'mfcc_3_delta_max_20ms',
		        'mfcc_4_delta_mean_20ms','mfcc_4_delta_std_20ms', 'mfcc_4_delta_min_20ms', 'mfcc_4_delta_max_20ms',
		        'mfcc_5_delta_mean_20ms','mfcc_5_delta_std_20ms', 'mfcc_5_delta_min_20ms', 'mfcc_5_delta_max_20ms',
		        'mfcc_6_delta_mean_20ms','mfcc_6_delta_std_20ms', 'mfcc_6_delta_min_20ms', 'mfcc_6_delta_max_20ms',
		        'mfcc_7_delta_mean_20ms','mfcc_7_delta_std_20ms', 'mfcc_7_delta_min_20ms', 'mfcc_7_delta_max_20ms',
		        'mfcc_8_delta_mean_20ms','mfcc_8_delta_std_20ms', 'mfcc_8_delta_min_20ms', 'mfcc_8_delta_max_20ms',
		        'mfcc_9_delta_mean_20ms','mfcc_9_delta_std_20ms', 'mfcc_9_delta_min_20ms', 'mfcc_9_delta_max_20ms',
		        'mfcc_10_delta_mean_20ms','mfcc_10_delta_std_20ms', 'mfcc_10_delta_min_20ms', 'mfcc_10_delta_max_20ms',
		        'mfcc_11_delta_mean_20ms','mfcc_11_delta_std_20ms', 'mfcc_11_delta_min_20ms', 'mfcc_11_delta_max_20ms',
		        'mfcc_12_delta_mean_20ms','mfcc_12_delta_std_20ms', 'mfcc_12_delta_min_20ms', 'mfcc_12_delta_max_20ms',
		        'mfcc_13_delta_mean_20ms','mfcc_13_delta_std_20ms', 'mfcc_13_delta_min_20ms', 'mfcc_13_delta_max_20ms',
		        'mfcc_1_mean_500ms','mfcc_1_std_500ms', 'mfcc_1_min_500ms', 'mfcc_1_max_500ms',
		        'mfcc_2_mean_500ms','mfcc_2_std_500ms', 'mfcc_2_min_500ms', 'mfcc_2_max_500ms',
		        'mfcc_3_mean_500ms','mfcc_3_std_500ms', 'mfcc_3_min_500ms', 'mfcc_3_max_500ms',
		        'mfcc_4_mean_500ms','mfcc_4_std_500ms', 'mfcc_4_min_500ms', 'mfcc_4_max_500ms',
		        'mfcc_5_mean_500ms','mfcc_5_std_500ms', 'mfcc_5_min_500ms', 'mfcc_5_max_500ms',
		        'mfcc_6_mean_500ms','mfcc_6_std_500ms', 'mfcc_6_min_500ms', 'mfcc_6_max_500ms',
		        'mfcc_7_mean_500ms','mfcc_7_std_500ms', 'mfcc_7_min_500ms', 'mfcc_7_max_500ms',
		        'mfcc_8_mean_500ms','mfcc_8_std_500ms', 'mfcc_8_min_500ms', 'mfcc_8_max_500ms',
		        'mfcc_9_mean_500ms','mfcc_9_std_500ms', 'mfcc_9_min_500ms', 'mfcc_9_max_500ms',
		        'mfcc_10_mean_500ms','mfcc_10_std_500ms', 'mfcc_10_min_500ms', 'mfcc_10_max_500ms',
		        'mfcc_11_mean_500ms','mfcc_11_std_500ms', 'mfcc_11_min_500ms', 'mfcc_11_max_500ms',
		        'mfcc_12_mean_500ms','mfcc_12_std_500ms', 'mfcc_12_min_500ms', 'mfcc_12_max_500ms',
		        'mfcc_13_mean_500ms','mfcc_13_std_500ms', 'mfcc_13_min_500ms', 'mfcc_13_max_500ms',
		        'mfcc_1_delta_mean_500ms','mfcc_1_delta_std_500ms', 'mfcc_1_delta_min_500ms', 'mfcc_1_delta_max_500ms',
		        'mfcc_2_delta_mean_500ms','mfcc_2_delta_std_500ms', 'mfcc_2_delta_min_500ms', 'mfcc_2_delta_max_500ms',
		        'mfcc_3_delta_mean_500ms','mfcc_3_delta_std_500ms', 'mfcc_3_delta_min_500ms', 'mfcc_3_delta_max_500ms',
		        'mfcc_4_delta_mean_500ms','mfcc_4_delta_std_500ms', 'mfcc_4_delta_min_500ms', 'mfcc_4_delta_max_500ms',
		        'mfcc_5_delta_mean_500ms','mfcc_5_delta_std_500ms', 'mfcc_5_delta_min_500ms', 'mfcc_5_delta_max_500ms',
		        'mfcc_6_delta_mean_500ms','mfcc_6_delta_std_500ms', 'mfcc_6_delta_min_500ms', 'mfcc_6_delta_max_500ms',
		        'mfcc_7_delta_mean_500ms','mfcc_7_delta_std_500ms', 'mfcc_7_delta_min_500ms', 'mfcc_7_delta_max_500ms',
		        'mfcc_8_delta_mean_500ms','mfcc_8_delta_std_500ms', 'mfcc_8_delta_min_500ms', 'mfcc_8_delta_max_500ms',
		        'mfcc_9_delta_mean_500ms','mfcc_9_delta_std_500ms', 'mfcc_9_delta_min_500ms', 'mfcc_9_delta_max_500ms',
		        'mfcc_10_delta_mean_500ms','mfcc_10_delta_std_500ms', 'mfcc_10_delta_min_500ms', 'mfcc_10_delta_max_500ms',
		        'mfcc_11_delta_mean_500ms','mfcc_11_delta_std_500ms', 'mfcc_11_delta_min_500ms', 'mfcc_11_delta_max_500ms',
		        'mfcc_12_delta_mean_500ms','mfcc_12_delta_std_500ms', 'mfcc_12_delta_min_500ms', 'mfcc_12_delta_max_500ms',
		        'mfcc_13_delta_mean_500ms','mfcc_13_delta_std_500ms', 'mfcc_13_delta_min_500ms', 'mfcc_13_delta_max_500ms']
		        
###############################################################
## 	         MAKE RELEVANT FILES. 		  	     ## 
###############################################################
os.chdir(cur_dir)
os.chdir(repo_name)
create_classifyfile(model_name)
create_YAML(repo_name)
create_dockercompose()
create_docker(incoming_topic, outgoing_topic)
create_makefile(repo_name, incoming_topic, outgoing_topic)
create_processpy(common_model_name)
create_readme(repo_name, classes, common_model_name, sampletype, giflink, default_features, standard_labels, modelinfo)
create_requirements()
create_server(common_model_name)
create_test(classes)
create_init1()
# copy over settings.json (in case you need defaults for some reason)
shutil.copy(prev_dir(cur_dir)+'/settings.json', os.getcwd()+'/settings.json')