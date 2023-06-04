# hash:sha256:6ee371a6e666bfdef6aed2ceb98d17d2b912d09a427b18055f3c61a17a558f9d
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    keras==2.9.0 \
    matplotlib==3.5.2 \
    numpy==1.22.4 \
    pandas==1.4.3 \
    seaborn==0.12.1 \
    tensorflow==2.9.1 \
    torch==2.0.0
