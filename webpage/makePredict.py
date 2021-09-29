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

def musicPrediction(data):
    model = joblib.load("MachineLearningModel/music-recommendation.joblib")
    prediction = model.predict(data)
    return prediction

def movieRecommendation(data):
    model = joblib.load()
    prediction = model.predict(data)
    pass
    return prediction
    
def breastCancerPrediction(data):
    model = joblib.load("MachineLearningModel/SVC-breast cancer.joblib")
    prediction = model.predict(data)
    return prediction

def animeRecommendation(data):
    model = joblib.load()
    prediction = model.predict(data)
    pass
    return prediction

def lotPricePrediction(data):
    model = joblib.load()
    prediction = model.predict(data)
    pass
    return prediction

def housePricePrediction(data):
    model = joblib.load()
    prediction = model.predict(data)
    pass
    return prediction