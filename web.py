# from flask import Flask
from flask.json import jsonify
from flask import request

from timeit import default_timer as timer
from datetime import datetime, date, timedelta



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


databaseURL = os.environ.get('MU_SPARQL_ENDPOINT', default="http://sem-eurod01.tenforce.com:8890/sparql")

@app.errorhandler(Exception)
def handle_invalid_usage(error):
    print(error)
    return error, 500

@app.route('/test')
@app.route('/test/')
@app.route('/test/<path:glob>')
def test(glob=None):
    week = request.args.get('week', "2017-05-22")
    publisher = request.args.get('publisher', 'demo')
    model = request.args.get('model', "RandomForest")

    try:
        print("Querying publisher <http://data.europa.eu/eurostat/id/organization/%(publisher)s> from week %(issued)s" % {'publisher': publisher, 'issued': week})
        start = timer()
        sparql = SPARQLWrapper(databaseURL + '?graph-realm-id=' + publisher)
        sparql.setQuery(sparqlPrefixes + """
            SELECT DISTINCT ?GTINdesc ?GTIN ?ISBA ?ISBAUUID ?ESBA ?ESBAdesc ?UUID ?quantity ?unit ?training
	        FROM <http://data.europa.eu/eurostat/temp>
            FROM <http://data.europa.eu/eurostat/ECOICOP>
	        WHERE {
                ?obs eurostat:product ?offer.
                ?offer a schema:Offer;
                    semtech:uuid ?UUID;
                    schema:description ?GTINdesc;
                    schema:gtin13 ?GTIN.
                OPTIONAL {
                    ?offer schema:includesObject [
                        a schema:TypeAndQuantityNode;
                        schema:amountOfThisGood ?quantity;
                        schema:unitCode ?unit
                    ].}
                OPTIONAL {
                    ?offer schema:category ?ISBA.
                    ?ISBA semtech:uuid ?ISBAUUID.
                    }
                ?obs eurostat:classification ?ESBA.
                ?ESBA skos:prefLabel ?ESBAdesc.
                ?obs qb:dataSet ?dataset.
                ?dataset dct:publisher <http://data.europa.eu/eurostat/id/organization/%(publisher)s>.
                ?dataset dct:issued "%(issued)s"^^xsd:dateTime.
                ?obs eurostat:training ?training.
            }
        """ % {'publisher': publisher, 'issued': week})
        sparql.method = 'POST'
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

        results = predict(training, production, model)

        return jsonify(results)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return jsonify({"error": repr(e)}), 500




@app.route('/classify')
@app.route('/classify/')
@app.route('/classify/<path:glob>')
def classify(glob=None):
    publisher = request.args.get('publisher', 'demo')
    week = request.args.get('week', "2017-05-22")
    model = request.args.get('model', "RandomForest")
    fromdate = str(datetime.strptime('2016-06-21', '%Y-%m-%d').date() - timedelta(days=365))

    try:
        print("Querying publisher <http://data.europa.eu/eurostat/id/organization/%(publisher)s> from week %(issued)s" % {'publisher': publisher, 'issued': week})
        # training data
        start = timer()
        sparql = SPARQLWrapper(databaseURL)
        sparql.setQuery(sparqlPrefixes + """
            SELECT DISTINCT ?GTINdesc ?GTIN ?ISBA ?ISBAUUID ?ESBA ?ESBAdesc ?UUID ?quantity ?unit ?training
	        FROM <http://data.europa.eu/eurostat/temp>
            FROM <http://data.europa.eu/eurostat/ECOICOP>
	        WHERE {
                ?obs eurostat:product ?offer.
                ?offer a schema:Offer;
                    semtech:uuid ?UUID;
                    schema:description ?GTINdesc;
                    schema:gtin13 ?GTIN.
                OPTIONAL {
                    ?offer schema:includesObject [
                        a schema:TypeAndQuantityNode;
                        schema:amountOfThisGood ?quantity;
                        schema:unitCode ?unit
                    ].}

                ?offer schema:category ?ISBA.
                ?ISBA semtech:uuid ?ISBAUUID.

                ?obs eurostat:classification ?ESBA.
                ?ESBA skos:prefLabel ?ESBAdesc.
                ?obs qb:dataSet ?dataset.
                ?dataset dct:publisher <http://data.europa.eu/eurostat/id/organization/%(publisher)s>.
                ?dataset dct:issued ?date.
                FILTER ( ?date >= "%(fromdate)s"^^xsd:dateTime).
                ?obs eurostat:training ?training.
            }
        """ % {'publisher': publisher, 'fromdate': fromdate})
        sparql.method = 'POST'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        training = []
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

            if record not in training:
                training.append(record)


        # production data
        start = timer()
        sparql = SPARQLWrapper(databaseURL + '?graph-realm-id=' + publisher)
        sparql.setQuery(sparqlPrefixes + """
            SELECT DISTINCT ?GTINdesc ?GTIN ?ESBA ?ESBAdesc ?UUID ?quantity ?unit
	        FROM <http://data.europa.eu/eurostat/temp>
            FROM <http://data.europa.eu/eurostat/ECOICOP>
	        WHERE {
                ?obs eurostat:product ?offer.
                ?offer a schema:Offer;
                    semtech:uuid ?UUID;
                    schema:description ?GTINdesc;
                    schema:gtin13 ?GTIN.
                OPTIONAL {
                    ?offer schema:includesObject [
                        a schema:TypeAndQuantityNode;
                        schema:amountOfThisGood ?quantity;
                        schema:unitCode ?unit
                    ].}

                ?obs eurostat:classification ?ESBA.
                ?ESBA skos:prefLabel ?ESBAdesc.
                ?obs qb:dataSet ?dataset.
                ?dataset dct:publisher <http://data.europa.eu/eurostat/id/organization/%(publisher)s>.
                ?dataset dct:issued "%(issued)s"^^xsd:dateTime.
                ?obs eurostat:training "false"^^<http://www.w3.org/2001/XMLSchema#boolean>.
            }
        """ % {'publisher': publisher, 'issued': week})
        sparql.method = 'POST'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        production = []
        for result in results["results"]["bindings"]:
            record = {}
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

            if record not in training:
                production.append(record)





        pprint(training)




        # Note that there can be overlap in the two sets.
        # Production data that already has a label will be reclassified,
        # but the given label will be in the response, too.
        data = training + production

        for row in data:
            desc = (" ").join(
                             row["ESBA-desc"].lower().split()
                             + row["GTIN-desc"].lower().split()
                             + row["unit"].lower().split()
                            )
            row["prod-desc"] = unicodedata.normalize('NFKD',desc).encode('ASCII', 'ignore').decode('utf-8')
        stop = timer()
        print("Queried in", stop - start)

        buildFeatureVectors(data)

        results = predict(training, production, model)

        return jsonify(results)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return jsonify({"error": repr(e)}), 500




























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

    return



def predict(training, production, modelName="RandomForest"):
    """
    Train the model on 'training' to make predictions for 'production'.
    Input:  'feat-vec'
    Output: 'ISBAUUID'
    """


    # Select the model.
    # Random forests is used in the prototype for interactive classification.
    # Logistic Regression is suggested for purely automatic classification.
    # Naive Bayes is a baseline reference.
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100)
        ,
        "LinearRegression": LR(multi_class="multinomial", solver="lbfgs")
        ,
        "NaiveBayes": MultinomialNB()
        }
    model = models[modelName]

    print("Model:", modelName)

    targetField = "ISBAUUID"

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
    if (not isba_labels) or (not key in isba_labels):
        print('Querying ISBA metadata')
        start = timer()
        sparql = SPARQLWrapper(databaseURL)
        sparql.setQuery("""
            PREFIX schema: <http://schema.org/>
            PREFIX offer: <http://data.europa.eu/eurostat/id/offer/>

            SELECT DISTINCT ?ISBA ?ISBAUUID ?ISBAdesc
	        FROM <http://data.europa.eu/eurostat/temp>
            WHERE {
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



sparqlPrefixes = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX qb: <http://purl.org/linked-data/cube#>
    PREFIX eurostat: <http://data.europa.eu/eurostat/ns/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX schema: <http://schema.org/>
    PREFIX sdmx-subject: <http://purl.org/linked-data/sdmx/2009/subject#>
    PREFIX sdmx-concept: <http://purl.org/linked-data/sdmx/2009/concept#>
    PREFIX sdmx-measure: <http://purl.org/linked-data/sdmx/2009/measure#>
    PREFIX interval: <http://reference.data.gov.uk/def/intervals/>
    PREFIX offer: <http://data.europa.eu/eurostat/id/offer/>
    PREFIX semtech: <http://mu.semte.ch/vocabularies/core/>"""
