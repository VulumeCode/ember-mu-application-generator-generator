from flask import Flask
from flask.json import jsonify
from flask import request

from timeit import default_timer as timer



import os
import csv
import re
import string
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation

from sklearn.externals import joblib

import unicodedata

plt.style.use(['dark_background'])
pd.set_option('display.max_colwidth', -1)

app = Flask(__name__)

@app.route('/mock')
def mock():
    data = readFromCSV('scannerdata/clean0329_ah.csv')

    split = int(len(data)*0.90)
    shuffle(data)
    training = data[:split]
    production = data[split:]

    # No cheating!
    for x in production:
        del x["ISBA-desc"]
        del x["ISBA"]
        del x["ECOICOP"]

    GTINvoc = buildVoc(data)

    buildFeatureVectors(data, GTINvoc)

    results = predict(training, production)
    return jsonify(results)

@app.route('/file')
def file():
    trainfilename = request.args.get('train', '')
    prodfilename  = request.args.get('prod', '')

    training = readFromCSV(trainfilename)
    production = readFromCSV(prodfilename)
    data = training + production

    GTINvoc = buildVoc(data)

    buildFeatureVectors(data, GTINvoc)

    results = predict(training, production)
    return jsonify(results)




def readFromCSV(filename):
    """
    Read a csv file and return a list of data pretty much like how we would get it from the DB
    """
    print("Loading", filename)

    data = []
    with open(filename, newline='', encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
                record = {}
                try: record["ECOICOP"] = row["ECOICOP"]
                except: pass
                try: record["ISBA"] = row["ISBA"]
                except: pass
                try: record["ISBA-desc"] = row["ISBA-description"].lower()
                except: pass
                try: record["ESBA"] = row["ESBA"]
                except: pass
                try: record["ESBA-desc"] = row["ESBA-description"].lower()
                except: pass
                try: record["GTIN"] = row["GTIN"]
                except: pass
                try: record["GTIN-desc"] = row["GTIN-description"].lower() + " " + row["UNIT"].lower()
                except: pass

                if record not in data:
                    data.append(record)
                else:
                    # print(record)
                    continue

    for row in data:
        desc = (" ").join(
                         row["ESBA-desc"].split()
                         + row["GTIN-desc"].split()
                        )
        row["prod-desc"] = unicodedata.normalize('NFKD',desc).encode('ASCII', 'ignore').decode('utf-8')

    return data




def buildVoc(data):
    """
    Takes a list of records with a field 'prod-desc'
    Returns a set of common substrings
    """
    # The vocabulary is all words in the dataset, longer than 2 chars
    GTINvoc = set()
    for x in data:
    #     GTINvoc |= set(x['prod-desc'].split())
        GTINvoc |= set(re.split(r"[0-9 -./]+", x['prod-desc']))

    # Add all possible len==3 ASCII strings
    GTINvoc |= {x+y+z for x in string.ascii_lowercase for y in string.ascii_lowercase for z in string.ascii_lowercase}

    GTINvoc = list(filter(lambda x: len(x) >= 2, GTINvoc))

    return GTINvoc




def buildFeatureVectors(data, voc):
    # Retrieve all stems from the voc in the string
    # Note: deduplicates words
    def stem(string, voc):
        result = []
        for stem in voc:
            if stem in string:
                result.append(stem)
        return result


    def pullGTINPrefix(gtin):
        digitStrings = []
        for idx, digit in enumerate(gtin):
            if idx < 10:
                digitStrings.append("p" + str(idx) + "_" + gtin[:idx+1])
        return digitStrings


    # Build feature string
    for row in data:
        row["feat-str"] = (" ").join(
                         ["ESBA"+row["ESBA"]]
                         + pullGTINPrefix(row["GTIN"])
                         + stem(row["prod-desc"], voc)
                        )

    # Build TFIDF transform
    docs = map(lambda x: x['feat-str'], data)

    # vectorizer = TfidfVectorizer(min_df=2, strip_accents="unicode").fit(docs)
    vectorizer = CountVectorizer(min_df=2, strip_accents="unicode").fit(docs)
    # vectorizer = HashingVectorizer(n_features=8192, norm='l2', strip_accents="unicode").fit(docs)

    for x in data:
        x['feat-vec'] = vectorizer.transform([x['feat-str']]).toarray()[0]

def predict(training, production):
    # Train the model
    targetField = "ISBA-desc"

    model = RandomForestClassifier(n_estimators=100) #, class_weight="balanced")

    # model = LR(multi_class="multinomial", solver="lbfgs")
    # model = LR()
    # model = LR(penalty="l1")
    # model = MultinomialNB(alpha=1)

    nrProductionPoints = len(production)
    nrTrainingPoints = len(training)
    nrPoints = nrProductionPoints + nrTrainingPoints
    nrFeatures = len(training[0]['feat-vec'])

    print("Nr products:", nrPoints)
    print("Used for training:", nrTrainingPoints)
    print("Used for testing:", nrProductionPoints)
    print("Nr features:", nrFeatures)

    trainingSample = np.zeros((nrTrainingPoints,nrFeatures))
    targets = []

    for idx_x, x in enumerate(training):
        targets.append(x[targetField])
        trainingSample[idx_x] = x['feat-vec']

    print("Nr classes:", len(set(targets)))

    start = timer()
    model.fit(trainingSample,targets)
    stop = timer()
    print("Trained in", stop - start)

    productionSample = np.zeros((nrProductionPoints,nrFeatures))

    for idx_x, x in enumerate(production):
        productionSample[idx_x] = x['feat-vec']

    def guessesTop(probs,classes):
        guesses = [{'isba_uuid': 1234, 'isba_label':isba, 'value':prob} for isba,prob in zip(classes, probs)]
        guesses = (sorted(guesses, key=lambda r:r['value']))[-5:]
        guesses.reverse()
        return guesses

    start = timer()
    probss = model.predict_proba(productionSample)
    stop = timer()
    print("Predicted in", stop - start)


    results = []
    for x, probs in zip(production, probss):
        top = guessesTop(probs,model.classes_)
        result = {
            "uuid":1234,
            "product":x['GTIN-desc'],
            "GTIN":x['GTIN'],
            "ESBA":x['ESBA'],
            "ESBA-desc":x['ESBA-desc'],
            "predictions":top
        }
        results.append(result)

    return results
