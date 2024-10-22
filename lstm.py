import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split
import pickle

import data


def build_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model

def run(max_epoch=25, nfolds=10, batch_size=128):
    indata = data.get_data()
    indata = list(indata)

    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    print(valid_chars)

    max_features = len(valid_chars) + 1 #合法的字符数
    maxlen = np.max([len(x) for x in X]) #最长的域名

    X = [[valid_chars[y] for y in x]for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

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
    filename = "model" + str(fold+1) + ".h5"
    model.save(filename)
    return final_data

if __name__ == "__main__":
    lstm_results = run(nfolds=1)
    print(lstm_results)
    results = {'lstm': lstm_results}
    RESULT_FILE = 'results.pkl'
    pickle.dump(results, open(RESULT_FILE, 'wb'))