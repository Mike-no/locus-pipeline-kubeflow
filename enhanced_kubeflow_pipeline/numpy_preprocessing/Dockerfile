FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3-pip python-dev

COPY ./requirements.txt /requirements.txt
COPY ./numpy_preprocessing.py /numpy_preprocessing.py

RUN pip3 install -r /requirements.txt