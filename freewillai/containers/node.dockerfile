FROM python:3.9-slim-buster AS node

# Force to save cache
ARG CACHEBUST=0

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install cmake wget file -y \
    && apt clean

COPY ./requirements.txt /app/requirements.txt
COPY ./freewillai/ /app/freewillai/
WORKDIR /node

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt
