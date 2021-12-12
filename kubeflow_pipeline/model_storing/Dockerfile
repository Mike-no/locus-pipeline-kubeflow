FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3-pip python-dev

COPY ./requirements.txt /requirements.txt
COPY ./minio_storing.py /minio_storing.py

RUN pip3 install -r /requirements.txt