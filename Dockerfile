FROM python:3.10.9-slim-buster

WORKDIR /

COPY . .

RUN pip install -r requirements.txt

RUN flask run Assignment_2/app.py