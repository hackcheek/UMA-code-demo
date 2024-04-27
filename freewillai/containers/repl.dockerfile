FROM python:3.10-slim-buster

COPY ./repl.requirements3.txt /repl/requirements.txt

RUN apt-get update && apt-get upgrade -y \
    && apt-get install \
        git \
        ca-certificates \
        ccache \
        cmake \
        curl \
        libgomp1 \
        libjpeg-dev \
        libpng-dev -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN pip install --upgrade pip \
    && pip install -r /repl/requirements.txt \
    # && pip install -U scikit-learn torch \
    && pip install git+https://github.com/hackcheek/tensorflow-onnx

COPY ./freewillai /repl/freewillai
COPY ./demos /repl/demos
COPY ./.env /repl/.env
COPY ./sepolia.env /repl/sepolia.env
COPY /demo_repl.env /repl/demo_repl.env

# ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

COPY ./scripts/exec_code.sh /repl/exec_code.sh

WORKDIR /repl

ENTRYPOINT ["sh", "exec_code.sh"]
# ENTRYPOINT ["python", "-c"]
