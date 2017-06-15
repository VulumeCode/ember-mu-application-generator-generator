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
from random import shuffle, random
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.externals import joblib
import unicodedata
import sys, traceback
from SPARQLWrapper import SPARQLWrapper, JSON


databaseURL = "http://localhost:8890/sparql"
# databaseURL = "http://sem-eurod01.tenforce.com:8890/sparql"


app = Flask(__name__)
    
@app.route('/test')
@app.route('/test/')
@app.route('/test/<path:supplier>')
def test(supplier=None):
    try:
        print('Querying')
        start = timer()
        sparql = SPARQLWrapper(databaseURL)
        sparql.setQuery("""
            prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            prefix qb: <http://purl.org/linked-data/cube#>
            prefix eurostat: <http://data.europa.eu/eurostat/ns/>
            prefix skos: <http://www.w3.org/2004/02/skos/core#>
            prefix dct: <http://purl.org/dc/terms/>
            prefix schema: <http://schema.org/>
            prefix sdmx-subject: <http://purl.org/linked-data/sdmx/2009/subject#>
            prefix sdmx-concept: <http://purl.org/linked-data/sdmx/2009/concept#>
            prefix sdmx-measure: <http://purl.org/linked-data/sdmx/2009/measure#>
            prefix interval: <http://reference.data.gov.uk/def/intervals/>
            prefix offer: <http://data.europa.eu/eurostat/id/offer/>
            prefix semtech: <http://mu.semte.ch/vocabularies/core/>

            select distinct ?GTINdesc ?GTIN ?ISBA ?ISBAUUID ?ESBA ?ESBAdesc ?UUID ?quantity ?unit ?training 
	    from <http://data.europa.eu/eurostat/temp>
            from <http://data.europa.eu/eurostat/ECOICOP>
	    where{
                ?obs eurostat:product ?offer.
                ?offer a schema:Offer;
                    semtech:uuid ?UUID;
                    schema:description ?GTINdesc;
                    schema:gtin13 ?GTIN.
                optional {
                    ?offer schema:includesObject [
                        a schema:TypeAndQuantityNode;
                        schema:amountOfThisGood ?quantity;
                        schema:unitCode ?unit
                    ].}
                optional {
                    ?offer schema:category ?ISBA.
                    ?ISBA semtech:uuid ?ISBAUUID.
                    }
                ?obs eurostat:classification ?ESBA.
                ?ESBA skos:prefLabel ?ESBAdesc.
                ?obs qb:dataSet ?dataset.
                ?dataset dct:publisher <http://data.europa.eu/eurostat/id/organization/demo>.
                ?obs eurostat:training ?training.
            }
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        data = []
        for result in results["results"]["bindings"]:
            record = {}
            try: record["ISBAUUID"] = result["ISBAUUID"]["value"]
            except: pass
            try: record["ESBA"] = re.search("/([0-9]+)$",result["ESBA"]["value"])[1]
            except: pass
            try: record["ESBA-desc"] = result["ESBAdesc"]["value"]
            except: pass
            try: record["GTIN"] = result["GTIN"]["value"]
            except: pass
            try: record["GTIN-desc"] = result["GTINdesc"]["value"]
            except: pass
            try: record["unit"] = result["unit"]["value"]
            except: record["unit"] = ""
            try: record["quantity"] = result["quantity"]["value"]
            except: record["quantity"] = ""
            try: record["UUID"] = result["UUID"]["value"]
            except: pass
            try: record["training"] = result["training"]["value"] == "1"
            except: pass

            if record not in data:
                data.append(record)
            else:
                # print(record)
                continue

        for row in data:
            desc = (" ").join(
                             row["ESBA-desc"].lower().split()
                             + row["GTIN-desc"].lower().split()
                             + row["unit"].lower().split()
                            )
            row["prod-desc"] = unicodedata.normalize('NFKD',desc).encode('ASCII', 'ignore').decode('utf-8')
        stop = timer()
        print("Queried in", stop - start)


        # Split the data in training and production data.
        # Note that there can be overlap in the two sets.
        # Production data that already has a label will be reclassified,
        # but the given label will be in the response, too.
        training = []
        production = []
        for row in data:
            if "ISBAUUID" in row:
                training.append(row)
            if not row["training"]:
                production.append(row)

        buildFeatureVectors(data)

        results = predict(training, production, "ISBAUUID")

        return jsonify(results)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return "{ error: " + str(e) + "}", 500
        

@app.route('/mock')
def mock():
    data = readFromCSV('scannerdata/clean0329_ah.csv')

    split = int(len(data)*0.90)
    shuffle(data)
    training = data[:split]
    production = data[split:]

    buildFeatureVectors(data)

    results = predict(training, production, "ISBA-desc")
    return jsonify(results)

@app.route('/csvfile')
def file():
    trainfilename = request.args.get('train', '')
    prodfilename  = request.args.get('prod', '')

    training = readFromCSV(trainfilename)
    production = readFromCSV(prodfilename)
    data = training + production

    buildFeatureVectors(data)

    results = predict(training, production, "ISBA-desc")
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
            try: record["ISBA-desc"] = row["ISBA-description"]
            except: pass
            try: record["ESBA"] = row["ESBA"]
            except: pass
            try: record["ESBA-desc"] = row["ESBA-description"]
            except: pass
            try: record["GTIN"] = row["GTIN"]
            except: pass
            try: record["GTIN-desc"] = row["GTIN-description"] + " " + row["UNIT"]
            except: pass
            record["UUID"] = "MOCKUUID"

            if record not in data:
                data.append(record)
            else:
                # print(record)
                continue

    for row in data:
        desc = (" ").join(
                         row["ESBA-desc"].lower().split()
                         + row["GTIN-desc"].lower().split()
                        )
        row["prod-desc"] = unicodedata.normalize('NFKD',desc).encode('ASCII', 'ignore').decode('utf-8')

    return data









def buildFeatureVectors(data):
    """
    Adds the feature vector 'feat-vec' to all records in data, using the other fields
    """

    def buildVoc(data):
        """
        Takes a list of records with a field 'prod-desc'
        Returns a vocabulary of common substrings
        The vocabulary is all words in the dataset, longer than 2 chars
        Add all possible ASCII-char 3-grams
        """
        GTINvoc = set()
        for x in data:
            GTINvoc |= set(re.split(r"[0-9 -./]+", x['prod-desc']))
        # 3-grams
        GTINvoc = set(filter(lambda x: len(x) >= 2, GTINvoc))
        GTINvoc |= {x+y+z for x in string.ascii_lowercase for y in string.ascii_lowercase for z in string.ascii_lowercase}
        return GTINvoc

    def stem(string, voc):
        """
        Retrieve all stems from the voc in the string
        Note: deduplicates words
        """
        result = []
        for stem in voc:
            if stem in string:
                result.append(stem)
        return result


    def pullGTINPrefix(gtin):
        """
        Decompose the GTIN string in its prefixes.
        For instance 8419700110360 becomes 8, 84, 841, 8419, 84197 and so on.
        """
        digitStrings = []
        for idx, digit in enumerate(gtin):
            if idx < 10:
                digitStrings.append("GTIN" + str(idx) + "_" + gtin[:idx+1])
        return digitStrings


    voc = buildVoc(data)

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

def predict(training, production, targetField):
    """
    Train the model on 'training' to make predictions for 'production'.
    Input:  'feat-vec'
    Target: targetField
    """
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

    def guessesTop(probs,classes, topN = 100):
        guesses =   [{'isba_uuid': isba,
                      'isba_label':isba_label(isba)["label"],
                      'notation':isba_label(isba)["notation"],
                      'value':prob}
                    for isba,prob in zip(classes, probs)]
        guesses = (sorted(guesses, key=lambda r:r['value']))[-topN:]
        guesses = list(filter(lambda r:r['value']>0, guesses))
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
            "uuid":     x['UUID'],
            "product":  x['GTIN-desc'],
            "GTIN":     x['GTIN'],
            "ESBA":     x['ESBA'],
            "ESBA-desc":x['ESBA-desc'],
            "unit":     x['unit'],
            "quantity": x['quantity'],
            "predictions":top
        }
        if "ISBAUUID" in x:
            result['classification'] = {
                "isba_uuid": x["ISBAUUID"],
                "isba_label": isba_label(x["ISBAUUID"])["label"],
                "notation": isba_label(x["ISBAUUID"])["notation"]
            }
        results.append(result)

    return results


isba_labels = {}
def isba_label(key):
    """
    Fetch and memoize isba labels.
    """
    global isba_labels
    if not isba_labels:
        print('Querying ISBA metadata')
        start = timer()
        sparql = SPARQLWrapper(databaseURL)
        sparql.setQuery("""
            prefix schema: <http://schema.org/>
            prefix offer: <http://data.europa.eu/eurostat/id/offer/>

            select distinct ?ISBA ?ISBAUUID ?ISBAdesc where{
                ?offer a schema:Offer;
                schema:category ?ISBA.
                ?ISBA skos:prefLabel ?ISBAdesc.
                ?ISBA <http://mu.semte.ch/vocabularies/core/uuid> ?ISBAUUID.
            }
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        isba_labels =   {result["ISBAUUID"]["value"]:
                            {"label":
                                result["ISBAdesc"]["value"]
                            , "notation":
                                re.search("/([0-9]+)$",result["ISBA"]["value"])[1]}
                        for result in results["results"]["bindings"]}
        stop = timer()
        print("Queried ISBA metadata in", stop - start)
    return isba_labels[key]

if __name__ == "__main__":
    app.run()
