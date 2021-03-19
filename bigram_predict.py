import keras
import os
from lstm_train import preprocessing
import numpy as np
import data
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import feature_extraction

def predict(domain,model_path="model1.h5"):
    model_path = os.path.join("saved_model",model_path)
    model = keras.models.load_model(model_path)
    domain = [domain] #转变为数组
    domain,max_features,maxlen = preprocessing(domain)
    res = model.predict(domain)
    print(res)


def run(force=False,model_path="bigram_model8.h5"):
    indata = data.get_data()
    indata = list(indata)

    # Extract data and labels
    X = [x[1] for x in indata] #域名
    labels = [x[0] for x in indata] #标签 

    # X,max_features,maxlen = preprocessing(X,force)
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    count_vec = ngram_vectorizer.fit_transform(X)


    y = [0 if x == 'benign' else 1 for x in labels] #目前是个二分类的问题

    final_data = []
    X_train, X_test, y_train, y_test, _, label_test = train_test_split(count_vec, y, labels, test_size=0.2) #train_test_split分割并打乱数据集
    model_path = os.path.join("saved_model",model_path)
    model = keras.models.load_model(model_path)
    res = model.predict_proba(X_test)
    res = [1 if x > 0.5 else 0 for x in res]
    print("precision=",sklearn.metrics.precision_score(y_test,res))
    print("recall=",sklearn.metrics.recall_score(y_test,res))
    print("fl-score=",sklearn.metrics.f1_score(y_test,res))
    print("accuracy=",sklearn.metrics.accuracy_score(y_test, res))
    print("auc=",sklearn.metrics.roc_auc_score(y_test,res))

run()
# predict("eclinicalweb.com")