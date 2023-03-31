FROM python:3.10.9-slim-buster

WORKDIR /

COPY requirements.txt requirements.txt
COPY Assignment_2/model /model
COPY Assignment_2/app.py /app.py

RUN pip install -r requirements.txt

ENTRYPOINT python app.py