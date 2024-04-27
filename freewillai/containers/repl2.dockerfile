FROM mambaorg/micromamba:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=1

COPY ./freewillai /repl/freewillai
COPY ./contracts /repl/contracts
COPY ./demos /repl/demos
COPY ./.env /repl/.env
COPY ./sepolia.env /repl/sepolia.env
COPY /demo_repl.env /repl/demo_repl.env
COPY ./scripts/exec_code.sh /repl/exec_code.sh
COPY ./scripts/exec_code.py /repl/exec_code.py

WORKDIR /repl
USER root

RUN rm -i /etc/apt/sources.list.d/*
RUN apt-get update && apt-get upgrade -y && apt-get install git -y && apt clean -y

RUN micromamba shell init --shell bash --root-prefix=~/micromamba \
    && eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate base \
    && micromamba install -c conda-forge python=3.10 -y \
    && micromamba install -c conda-forge -y pytorch scikit-learn=1.2 web3 pip \
        tqdm onnx onnxruntime pandas polars transformers pillow torchvision \
        python-dotenv base58 py-solc-x skl2onnx tensorflow \
    && pip install aioipfs \
    && pip install git+https://github.com/hackcheek/tensorflow-onnx

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-u", "./exec_code.py"]
