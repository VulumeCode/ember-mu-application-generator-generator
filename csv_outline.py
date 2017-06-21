





# Reading from a CSV file and using data from flat files is a bit different from using triple stores.
# Most changes are trivial, and would be specific to your implementation anyways.
# However, since we started out with a CSV based implementation, here are some snippets that might help you out, detailing the biggest changes.





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













def predict(training, production):
    """
    Train the model on 'training' to make predictions for 'production'.
    Input:  'feat-vec'
    Output: 'ISBA-desc'
    """
    # Train the model
    targetField = "ISBA-desc"

    # Select the model.
    # Random forests is used in the prototype for interactive classification.
    # Logistic Regression is suggested for purely automatic classification.
    # Naive Bayes is a baseline reference.
    model = RandomForestClassifier(n_estimators=100)
    # model = LR(multi_class="multinomial", solver="lbfgs")
    # model = MultinomialNB()

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
