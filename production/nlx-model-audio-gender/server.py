import os
import json
from kafka import KafkaConsumer, KafkaProducer
import process
import requests

# configure kafka
KAFKA_HOST = os.environ['KAFKA_HOST']
KAFKA_INCOMING_TOPIC = os.environ['KAFKA_INCOMING_TOPIC']
KAFKA_OUTGOING_TOPIC = os.environ['KAFKA_OUTGOING_TOPIC']
OBJECT_ENDPOINT = os.environ['OBJECT_ENDPOINT']
bootstrap_servers = [KAFKA_HOST]
incoming_topic = KAFKA_INCOMING_TOPIC
outgoing_topic = KAFKA_OUTGOING_TOPIC
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
consumer = KafkaConsumer(incoming_topic, bootstrap_servers=bootstrap_servers)
print(f'Microservice listening to topic: {incoming_topic}...')
# iterate on the samples
# ex: {"id": "NLX-1520555910458794813-1520555959992", "type": "v1Sample"}
for msg in consumer:
  # output message
  print(f'New message: {msg.value}')
  # read the payload and process
  payload = json.loads(msg.value)
  # fix issues with poorly formed payloads
  if 'type' not in payload:
    payload['type'] = 'sample'
  if 'id' not in payload:
    payload['id'] = payload['sampleId']
  sampleId = payload['id']
  print(payload)
  # process the payload
  process.process(payload)
  # asynchronously send the status
  try:
    print('Submitting status for sample: ' + sampleId)
    r = requests.post(OBJECT_ENDPOINT + '/status/samples/' + sampleId + '/modeled-gender')
  except:
    print('Failed to submit status for sample: ' + sampleId)
  # send sample to the next topic
  print(f'Submitting message on {outgoing_topic}')
  producer.send(outgoing_topic, msg.value)
