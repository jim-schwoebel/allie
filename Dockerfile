FROM ubuntu:latest

WORKDIR /usr/src/app
COPY . /usr/src/app

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y tree \
  && apt-get install -y libsm6 \ 
  && apt-get install libsndfile1

RUN     python setup.py
RUN		tree
WORKDIR /usr/src/app/allie
