from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import subprocess
import csv
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

def score(text, model, threshold = 0.5):
    subprocess.Popen('mlflow ui', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    best_model = mlflow.sklearn.load_model(model)
    messages = pd.read_csv('data/train.csv', sep='\t', quoting=csv.QUOTE_NONE, index_col=False)
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    messages_bow = bow_transformer.transform(messages['message'])
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    valid_bow = bow_transformer.transform([text])
    valid_tfidf = tfidf_transformer.transform(valid_bow)
    propensity = best_model.predict_proba(valid_tfidf)
    prediction = propensity[0,1] > threshold
    subprocess.Popen('pkill -f gunicorn', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return prediction, propensity[0,1]
