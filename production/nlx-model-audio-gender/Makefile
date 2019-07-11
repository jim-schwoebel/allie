build:
	docker build . -t nlx-model-audio-gender
run:
	docker run -d \ 
		--network=nlx-model-audio-gender_default \ 
		--name=nlx-model-audio-gender_worker_1 \ 
		-e "KAFKA_HOST=kafka-service:9092" \ 
		-e "KAFKA_INCOMING_TOPIC=REQUESTED_MODEL_GENDER" \ 
		-e "KAFKA_OUTGOING_TOPIC=CREATED_MODEL_GENDER" \ 
		-e "MONGO_URL=mongo-service:27017" \ 
		-e "MONGO_DB=nlx-data" \ 
		nlx-model-audio-gender
stop:
	docker stop nlx-model-audio-gender_worker_1 || true
	docker-compose stop
kill: stop
	docker rm nlx-model-audio-gender_worker_1
	docker-compose down
up:
	docker-compose up -d
	printf 'Waiting for kafka...\n' 
	sleep 15
	make setup
	make build
	make run
down:
	make kill
setup:
	docker exec -it nlx-model-audio-gender_kafka-service_1 \ 
		/usr/bin/kafka-topics \ 
		--create \ 
		--zookeeper zookeeper-service:2181 \ 
		--replication-factor 1 \ 
		--partitions 1 \ 
		--topic REQUESTED_MODEL_GENDER
	docker exec -it nlx-model-audio-gender_kafka-service_1 \ 
		/usr/bin/kafka-topics \ 
		--create \ 
		--zookeeper zookeeper-service:2181 \ 
		--replication-factor 1 \ 
		--partitions 1 \ 
		--topic CREATED_MODEL_GENDER 
# populate:
	# docker exec -it nlx-model-audio-gender_kafka-service_1 /usr/bin/kafka-console-producer --broker-list kafka-service:9092 --topic REQUESTED_MODEL_GENDER
	# > {"id":""}
