import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def min_max_scaler(np):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np)
    return scaled
