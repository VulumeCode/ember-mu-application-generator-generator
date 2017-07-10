Classificaton backend
===

Trains a model and classifies data from the triplestore.

Query parameters
---

Data is selected with its `week` (indicated by a Monday "YYYY-MM-DD") and `publisher` id (the retail chain).

The `model` can be one of: "RandomForest", "LinearRegression", "NaiveBayes". The default is RandomForest.

Endpoints
---

Training data is the labeled data which the model is trained on.

Production data is the data which is classified by the model.

If production data already has labels, the assigned label is returned alongside the predictions.

Data selection differs per endpoint:

`/test`: Training data is all labeled data. Production data is all data without the `eurostat:training "True"` predicate.

`/classify`: Training data is all labeled data with the `eurostat:training "True"` predicate, and all labeled data from up to a year before the selected week. Production data is all data from the selected week. 
