import numpy as np # for working with arrays
import pandas as pd # for working with data set, data frame structure
import yfinance as yf # for using Yahoo finance API
import matplotlib.pyplot as plt # for data visualization

# build and train everything with the neural network
import torch 
import torch.nn as nn
import torch.optim as optim


from sklearn.preprocessing import StandardScaler # for scaling data
from sklearn.metrics import root_mean_squared_error # for evaluating our model

from kafka import KafkaProducer

import json
import time

if __name__ == "__main__":
    # Use GPU for processing large tensors and matrix operations if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    ticker = 'AAPL' # ticker for Apple

    # uses the yfinance library to download historical stock price data for Apple Inc. starting from January 1, 2024 up to the current date
    df = yf.download(ticker, '2020-01-01')
    # print(df)

    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

    for _, row in df.iterrows():
        message = row.to_dict()
        producer.send('stock_prices', value=message)
        print(f"Sent: {message}")
        time.sleep(1)  # send one row per second (simulate streaming)