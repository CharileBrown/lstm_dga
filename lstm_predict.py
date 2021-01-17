import keras
import os

def dealInput(domain):
    

def predict(model_path="model10.h5"):
    model_path = os.path.join("saved_model",model_path)
    model = keras.models.load_model(model_path)