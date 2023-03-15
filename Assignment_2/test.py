from score import score
import numpy as np
import requests
import json
import subprocess
import os

def test_score():
    #Obvious spam
    ip = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    model = "runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69"
    threshold = 0.7
    prediction, propensity = score(ip, model, threshold)
    print(prediction, propensity)
    assert prediction == True
    
    #Smoke test
    assert prediction != None
    assert propensity != None

    #Format test
    assert type(propensity) == np.float64
    assert type(prediction) == np.bool_

    #Make sure the values fall in the expected domain of values

    assert prediction == True or prediction == False
    assert propensity >= 0 and propensity <= 1

    #Threshold of 0 always labels spam
    prediction, propensity = score(ip, model, 0)
    assert prediction == True

    #Threshold of 1 always labels not spam
    prediction, propensity = score(ip, model, 1)
    assert prediction == False

    #Obvious not spam
    prediction, propensity = score("hi, wassup?", model, 0.5)
    assert prediction == False

def test_flask():
    os.system('python app.py &')
    text = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    y = requests.post("http://127.0.0.1:8080/", json = {"text": text})
    print(y.json())


test_score()
test_flask()
