# Build image with: 
# -> docker build -t allie_image .
# read more @ https://github.com/jim-schwoebel/allie/wiki/6.-Using-Allie-and-Docker

# get Ubuntu base image
FROM	ubuntu:18.04 AS base
MAINTAINER Jim Schwoebel <jim.schwoebel@gmail.com>

# set working directory
WORKDIR /usr/src/app
COPY . /usr/src/app

# set environment variables 
ENV DEBIAN_FRONTEND=noninteractive 

# now run sudo apt update commands
RUN apt-get update \
  && apt-get install -y python3-pip python3 \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y apt-utils \
  && apt-get install -y autoconf \
  && apt-get install -y automake \
  && apt-get install -y build-essential \
  && apt-get install -y cmake \
  && apt-get install -y ffmpeg \
  && apt-get install -y gcc \
  && apt-get install -y g++ \
  && apt-get install -y git \
  && apt-get install -y libasound-dev \
  && apt-get install -y libffi-dev \
  && apt-get install -y libldap2-dev \
  && apt-get install -y libpq-dev \
  && apt-get install -y libpulse-dev \
  && apt-get install -y libsasl2-dev \
  && apt-get install -y libsm6 \ 
  && apt-get install -y libsndfile1 \
  && apt-get install -y libtool \
  && apt-get install -y libxml2-dev \
  && apt-get install -y libxslt1-dev \ 
  && apt-get install -y make \ 
  && apt-get install -y m4 \
  && apt-get install -y opus-tools \
  && apt-get install -y portaudio19-dev \
  && apt-get install -y sox \
  && apt-get install -y swig \
  && apt-get install -y tesseract-ocr \
  && apt-get install -y tree \
  && apt-get install -y tzdata \
  && apt-get install -y unzip \
  && apt-get install -y wget

# install openSMILE
RUN	wget https://www.audeering.com/download/opensmile-2-3-0-tar-gz/\?wpdmdl\=4782 -O opensmile-2.3.0.tar.gz&& \
	tar -xf opensmile-2.3.0.tar.gz -C /usr/local 
RUN    cd /usr/local/opensmile-2.3.0/ && \
	sed -i '117s/(char)/(unsigned char)/g' src/include/core/vectorTransform.hpp && \
	./buildWithPortAudio.sh -o /usr/local/lib && \
	./buildStandalone.sh -o /usr/local/lib
RUN	chmod 777 /usr/local/opensmile-2.3.0 
RUN	export PATH=/usr/local/opensmile-2.3.0/inst/bin:$PATH

# install requirements
RUN	pip3 install numpy==1.18.2
RUN	pip3 install -r requirements.txt

# order matters here for OpenCV and a few of the dependencies below
# custom pip3 installations across all operating systems
RUN	pip3 install --upgrade mxnet pytest==5.4.3 scipy==1.4.1 scikit-learn==0.22.2.post1 \
  && pip3 install git+https://github.com/detly/gammatone.git \
  && pip3 install https://github.com/vBaiCai/python-pesq/archive/master.zip \
  # && pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu] \
  && pip3 install tsaug==0.2.1 \
  && pip3 install psutil==5.7.2 \
  && pip3 install pyfiglet==0.8.post1 \
  && pip3 install gensim==3.8.3 \
  && pip3 install wget==3.2 \
  && pip3 install textblob==0.15.3 \
  && pip3 install moviepy==1.0.3 \
  && pip3 install textacy==0.10.0 \
  && pip3 install SpeechRecognition==3.8.1 \
  && pip3 install pytesseract==0.3.4 \
  && pip3 install pydub==0.24.1 \
  && pip3 install ctgan==0.2.1 \
  && pip3 install librosa==0.6.2 \
  && pip3 install sk-video==1.1.10 \
  && pip3 install opencv-python==3.4.2.17 \
  && pip3 install opencv-contrib-python==3.4.2.17 \
  && pip3 install nltk==3.4.5 \
  && pip3 install umap-learn==0.4.6 \
  && pip3 install numba==0.48 \
  && python3 -m spacy download 'en' \
  && python3 -m spacy download 'en_core_web_sm'

# now run unit tests
WORKDIR	/usr/src/app

# now download required NLTK dependencies and run unit tests
RUN     python3 docker.py

# now open up CLI with 
# docker run -it --entrypoint=/bin/bash allie_image