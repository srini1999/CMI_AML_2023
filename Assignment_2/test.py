from score import score
import numpy as np
import requests
import time
import os
import pytest


@pytest.fixture
def params():
    ip = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    model = "runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69"
    threshold = 0.7
    prediction, propensity = score(ip, model, threshold)
    return [prediction, propensity]

def test_spam(params):
    assert params[0] == True

def test_smoke(params):
    assert params[0] != None
    assert params[1] != None

def test_format(params):
    assert type(params[1]) == np.float64
    assert type(params[0]) == np.bool_

#Make sure the values fall in the expected domain of values
def test_dom(params):
    assert params[0] == True or params[0] == False
    assert params[1] >= 0 and params[1] <= 1

#Threshold of 0 always labels spam
def test_sanity1():
    ip = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    model = "runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69"
    prediction, propensity = score(ip, model, 0)
    assert prediction == True

#Threshold of 1 always labels not spam
def test_sanity2():
    ip = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    model = "runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69"
    prediction, propensity = score(ip, model, 1)
    assert prediction == False

#Obvious not spam
def test_ham():
    model = "runs:/5c18105aa4f34a6188cf0da1dbd3f962/SVM_69"
    prediction, propensity = score("hi, wassup?", model, 0.5)
    assert prediction == False

def test_flask():
    os.system('python app.py>/dev/null 2>&1 &')
    time.sleep(5)
    text = "Sure Shot INTRADAY & MULTIBAGGER Stock Tips - Earn 120% PROFIT in 4 Month https://bit.ly/NSE_7 - Click on Link & Send 'JOIN FREE' Message on WhatsApp EXPTRADE"
    y = requests.post("http://127.0.0.1:8080/", json = {"text": text})
    y = dict(y.json())
    assert y["prediction"] == 'spam'


