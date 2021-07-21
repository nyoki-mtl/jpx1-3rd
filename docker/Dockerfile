FROM continuumio/anaconda3:2019.03

RUN apt-get update && apt-get install -y zip gosu

ADD docker/requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

ADD docker/entrypoint.sh /tmp
ENTRYPOINT ["bash", "/tmp/entrypoint.sh"]
