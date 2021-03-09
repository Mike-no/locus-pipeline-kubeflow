FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3-pip python-dev

COPY ./requirements.txt /requirements.txt
COPY ./pandas_loading.py /pandas_loading.py
COPY ./e1.csv /e1.csv

RUN pip3 install -r /requirements.txt
