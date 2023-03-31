# syntax=docker/dockerfile:1

# For power pc architecture use: ppc64le/ubuntu:22.04
ARG PLATFORM=amd64
FROM --platform=$PLATFORM ubuntu:22.04

WORKDIR /workspace/dependencies

RUN apt-get update
RUN apt-get -y install openjdk-19-jdk
RUN apt-get -y install ant
RUN apt-get -y install python3-minimal
RUN apt-get -y install python3.10-full
RUN apt-get -y install python3-pip
RUN apt-get -y install git
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_lg
RUN ls -la
ENV MALLET_HOME=./Mallet/bin
RUN mkdir tmp
ENV NUMBA_CACHE_DIR=./tmp
RUN mkdir matplotlib_tmp
ENV MPLCONFIGDIR=./matplotlib_tmp
RUN mkdir -p huggingface/hub
ENV TRANSFORMERS_CACHE=./huggingfache/hub
RUN pip3 list