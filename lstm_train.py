import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split
import pickle
import os

import data


def build_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen)) #用词向量来表示域名
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model

def preprocessing(X,force=False):
    Data_Feature_FILE = "dataFeature.pkl"
    dataFeature = ''
    if force or (not os.path.isfile(Data_Feature_FILE)):
        valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
        max_features = len(valid_chars) + 1 #合法的字符数
        maxlen = np.max([len(x) for x in X]) #最长的域名
        pickle.dump((valid_chars,max_features,maxlen),open(Data_Feature_FILE,"wb"))
    else:
        dataFeature = pickle.load(open(Data_Feature_FILE,"rb+"))
    valid_chars,max_features,maxlen = dataFeature[0],dataFeature[1],dataFeature[2]
    
    # print(valid_chars,max_features,maxlen)

    X = [[valid_chars[y] for y in x]for x in X] # X每个元素是将域名映射字符与数字一一映射后的数组
    X = sequence.pad_sequences(X, maxlen=maxlen) # 将数组补齐至最长域名的长度
    '''
        X的元素像这样:
        [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        38 32 14 18 15 17 32 36 28 22 25 12  2 24 25 23 29 35 15  9  6 17 34 14
        11 23 16  9]
    '''
    # debug代码,打印x的信息
    # for i in range(1,10):
    #     print(X[i],len(X[i]))

    # exit()
    return X,max_features,maxlen

'''
    模型一次训练batch_size个数据
    模型将整个数据集训练max_epoch次
'''
def run(max_epoch=256, nfolds=10, batch_size=4096,force=False):
    indata = data.get_data()
    indata = list(indata)

    # Extract data and labels
    X = [x[1] for x in indata] #域名
    labels = [x[0] for x in indata] #标签 

    X,max_features,maxlen = preprocessing(X,force)

    y = [0 if x == 'benign' else 1 for x in labels] #目前是个二分类的问题

    final_data = []

    for fold in range(nfolds):
        print ("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2) #train_test_split分割并打乱数据集

        print ("Build model...")
        model = build_model(max_features, maxlen)

        print("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05) 
        best_iter = -1
        best_auc = 0.0
        out_data = {}
        
        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)
        
            print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))
            
            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)

                tmp_probs = [1 if x > 0.5 else 0 for x in probs]

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_martix': sklearn.metrics.confusion_matrix(y_test, tmp_probs)}
                
                
                print(sklearn.metrics.confusion_matrix(y_test, tmp_probs)) 
            else:
                if(ep-best_iter) > 2:
                    break

            final_data.append(out_data)
        filename = "saved_model/model" + str(fold+1) + ".h5"
        model.save(filename)
    return final_data

if __name__ == "__main__":
    lstm_results = run()
    results = {'lstm': lstm_results}
    RESULT_FILE = 'results.pkl'
    pickle.dump(results, open(RESULT_FILE, 'wb'))
