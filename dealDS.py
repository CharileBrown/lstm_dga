import os
import sys
import csv
import pickle

DATA_FILE = 'traindata.pkl'

def get_malicious(maxNum=1000): #生成恶意域名数据集,一共生成了23个类
    dga = open("dga.txt","r")
    data = open("data.csv","w",newline='')
    writer = csv.writer(data)
    cls_map = {} #计算每个类的数量
    cls_need = {}  #存储需要哪些class
    while True:
        line = dga.readline()
        if not line:
            break
        args = line.split('\t')
        label, domain = args[0], args[1]
        if label not in cls_map:
            cls_map[label] = 1
        else:
            if cls_map[label] >= maxNum:
                continue
            cls_map[label]+=1
        print(label,domain,cls_map[label])
    print(cls_map)
    for key,value in cls_map.items():
        if value == 1000:
            cls_need[key] = 0
    dga = open("dga.txt","r")
    while True:
        line = dga.readline()
        if not line:
            break
        args = line.split('\t')
        label, domain = args[0], args[1]
        if label not in cls_need:
            continue
        else:
            if cls_need[label] >= maxNum:
                continue
            cls_need[label]+=1
            writer.writerow((label,domain))
    dga.close()
    data.close()

def get_alexa(): #生成正经域名数据集
    alexa = open("alexa.csv","r")
    data = open("data.csv","a",newline='')
    writer = csv.writer(data)
    cnt = 20000
    while cnt:
        line = alexa.readline()
        domain = line.split(",")[1].strip()
        print("benign",domain)
        writer.writerow(("benign",domain))
        cnt-=1
    alexa.close()
    data.close()

def gen_data(force=False):
    if force or (not os.path.isfile(DATA_FILE)):
        dataset = open("data.csv","r")
        domains = []
        labels = []
        while True:
            line = dataset.readline().strip()
            if not line:
                break
            args = line.split(",")
            domains += [args[1]]
            labels += [args[0]]
        pickle.dump(zip(labels, domains), open(DATA_FILE, 'wb'))
    
def get_data(force=False):
    gen_data(force)
    return pickle.load(open(DATA_FILE,"rb+"))

# get_malicious()
# get_alexa()
