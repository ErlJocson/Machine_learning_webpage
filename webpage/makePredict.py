from pandas import DataFrame
import joblib

def heartAttackPrediction(data):
    model = joblib.load("MachineLearningModel/Model-svm.joblib")
    prediction = model.predict(data)
    return prediction

def lungCancerPrediction(data):
    model = joblib.load("MachineLearningModel/logit-lung-cancer.joblib")
    prediction = model.predict(data)
    return prediction

def titanicDataset(data):
    model = joblib.load("MachineLearningModel/titanic.joblib")
    prediction = model.predict(DataFrame(data, columns= ["Pclass","Sex","Embarked"]))
    return prediction