# hash:sha256:2fcfb84cda9528b46bb89b0a6080c486ddad5f47d1d693c06e6a9c84280e2755
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    jupyter==1.0.0 \
    matplotlib==3.7.1 \
    numpy==1.24.3 \
    scikit-learn==1.2.2 \
    tensorflow==2.12.0
