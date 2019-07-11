FROM gcr.io/arctic-robot-192514/nlx-base
WORKDIR /usr/src/app
ADD . /usr/src/app
# install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# run tests
RUN python test.py
# env
ENV KAFKA_HOST="kafka-service:9092"
ENV KAFKA_INCOMING_TOPIC="REQUESTED_MODEL_GENDER"
ENV KAFKA_OUTGOING_TOPIC="CREATED_MODEL_GENDER"
ENV MONGO_URL="mongo-service:27017"
ENV MONGO_DB="nlx-data"
CMD ["python", "-u", "/usr/src/app/server.py"]