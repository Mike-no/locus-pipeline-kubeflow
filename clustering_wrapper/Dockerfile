FROM python:3.7-slim

COPY requirements.txt /clustering/requirements.txt
COPY ClusteringModel.py /clustering/ClusteringModel.py

WORKDIR /clustering

RUN pip install -r requirements.txt

EXPOSE 5000

USER root
RUN chmod 777 /clustering
USER 8888

CMD exec seldon-core-microservice ClusteringModel --service-type MODEL