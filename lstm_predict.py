import keras
import os
from lstm_train import preprocessing
import numpy as np
import data
from sklearn.model_selection import train_test_split
import sklearn

def predict(domain,model_path="model10.h5"):
    model_path = os.path.join("saved_model",model_path)
    model = keras.models.load_model(model_path)
    domain = [domain] #转变为数组
    domain,max_features,maxlen = preprocessing(domain)
    res = model.predict_classes(domain)
    print(res)


def run(force=False,model_path="model1.h5"):
    indata = data.get_data()
    indata = list(indata)

    # Extract data and labels
    X = [x[1] for x in indata] #域名
    labels = [x[0] for x in indata] #标签 

    X,max_features,maxlen = preprocessing(X,force)

    y = [0 if x == 'benign' else 1 for x in labels] #目前是个二分类的问题

    final_data = []
    X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2) #train_test_split分割并打乱数据集
    model_path = os.path.join("saved_model",model_path)
    model = keras.models.load_model(model_path)
    res = model.predict(X_test)
    res = [1 if x > 0.5 else 0 for x in res]
    print(sklearn.metrics.accuracy_score(y_test, res))

run()
# predict("ultraporader-conapefy-prolobeziless.info")