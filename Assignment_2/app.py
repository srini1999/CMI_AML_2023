from flask import Flask, request
import mlflow
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import csv
import pandas as pd

app = Flask(__name__)

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

@app.route('/', methods = ['POST','GET'])
def index():
    if request.method == 'POST':

        best_model = mlflow.sklearn.load_model("runs:/a9bfc75b3fcd461ba87ef42650c6f4f3/SVM_69")
        messages = pd.read_csv('data/train.csv', sep='\t', quoting=csv.QUOTE_NONE, index_col=False)
        bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])  
        messages_bow = bow_transformer.transform(messages['message'])
        tfidf_transformer = TfidfTransformer().fit(messages_bow)  
        ip_bow = bow_transformer.transform(ip)
        ip_tfidf = tfidf_transformer.transform(ip_bow)
    else:
    
        return "Hello"

if __name__ == "__main__":
    app.run() 