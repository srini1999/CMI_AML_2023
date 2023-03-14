from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import numpy as np
import csv
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


best_model = mlflow.sklearn.load_model("runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69")
messages = pd.read_csv('data/train.csv', sep='\t', quoting=csv.QUOTE_NONE, index_col=False)
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
messages_bow = bow_transformer.transform(messages['message'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
ip = ["Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"]
valid_bow = bow_transformer.transform(ip)
valid_tfidf = tfidf_transformer.transform(valid_bow)
print(best_model.predict_proba(valid_tfidf))