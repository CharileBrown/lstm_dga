from io import StringIO, BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pickle
import os
import random
import tldextract
import datetime


ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Our ourput file containg all the training data
DATA_FILE = 'traindata.pkl'

def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
    """Grabs Alexa 1M"""
    url = urlopen(address)
    zipfile = ZipFile(BytesIO(url.read()))
    # return [tldextract.extract(x.split(',')[1]).domain for x in \
    #         zipfile.read(filename).split()[:num]]
    print( zipfile.read(filename).split()[:num] )
    domain = [bytes.decode(x) for x in zipfile.read(filename).split()[:num]]
    print([tldextract.extract(x.split(',')[1]).domain for x in domain])

get_alexa(num=10)  