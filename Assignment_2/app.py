from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
from textblob import TextBlob
import pickle
import csv
# from score import score
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def index():
    text = request.json['text']
    messages = pd.read_csv('data/train.csv', sep='\t', quoting=csv.QUOTE_NONE, index_col=False)
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    messages_bow = bow_transformer.transform(messages['message'])
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    valid_bow = bow_transformer.transform([text])
    valid_tfidf = tfidf_transformer.transform(valid_bow)
    model = pickle.load(open("model/model.pkl","rb"))
    pred = model.predict(valid_tfidf)[0]
    return json.dumps({"prediction": pred})

if __name__ == "__main__":
    app.run(debug = True)