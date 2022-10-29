#!/bin/bash
# backend/Dockerfile

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
#FROM nvidia/cuda:11.2.1-base-ubuntu18.04

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /backend/requirements.txt

WORKDIR /backend

RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

COPY . /backend

EXPOSE 8000
#127.0.0.1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]
#CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000" , "--reload"]