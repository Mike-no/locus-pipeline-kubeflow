FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3-pip python-dev
RUN mkdir ./models

COPY ./models_keeper.py /models_keeper.py
COPY ./data/e1.csv ./models/e1.csv
COPY ./data/nodes.npz ./models/nodes.npz
COPY ./data/tokyo_osmpbf.ft ./models/tokyo_osmpbf.ft
COPY ./data/tree.pkl ./models/tree.pkl
