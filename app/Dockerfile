FROM ubuntu:latest

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install streamlit
RUN pip3 install scikit-learn