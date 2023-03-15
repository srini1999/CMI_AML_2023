from flask import Flask, request
import json
from score import score

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def index():
    text = request.json['text']
    pred, propensity = score(text)
    if pred == True:
        prediction = "spam"
    else:
        prediction = "ham"
    return json.dumps({"prediction": prediction, "propensity": propensity})

if __name__ == "__main__":
    app.run(debug = True, port = 8080)