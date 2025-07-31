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

from kafka import KafkaConsumer

import json
import time
from datetime import datetime
import os

from model import PredictModel

# calculate the train and test Root Mean Squared Error 
def manual_rmse(y_true, y_pred):
    # Ensure both tensors are on the same device
    diff = y_pred - y_true  # difference between predictions and actuals
    squared_diff = diff ** 2  # square the differences
    mean_squared_error = torch.mean(squared_diff)  # take the average
    rmse = torch.sqrt(mean_squared_error)  # square root of the average
    return rmse

consumer = KafkaConsumer(
    "stock_data",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Consumer is listening...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for message in consumer:
    print(message)
    data = message.value
    ticker = data["ticker"]
    prices = pd.DataFrame(data["prices"], columns=["Close"])

    # scale all of the data to fit distribution
    scaler = StandardScaler()
    prices["Close"] = scaler.fit_transform(prices[["Close"]])

    # prepare the data for out neural network
    # we want at any point and time to look at the past x days, where x is some arbitrary integer, and predict what the stock price will be for the next day (next data point)

    seq_length = 30 # how many days we would like to consider
    seq_data = []
    seq_dates = []
    for i in range(len(prices) - seq_length):
        seq_data.append(prices["Close"].values[i:i+seq_length])
        seq_dates.append(prices.index[i+seq_length])

    if not seq_data:
        print("Not enough data to make a prediction.")
        continue

    seq_data = np.array(seq_data)
    seq_data = np.expand_dims(seq_data, axis=-1)  # shape: (num_sequences, seq_length, 1)

    # train the first 80% of the data (use for input) for sequential predicting
    train_size = int(0.8 * len(seq_data)) 

    X_train = torch.tensor(seq_data[:train_size, :-1, :]).type(torch.Tensor).to(device) # 1st dimension (outter most), 2nd dimension, 3rd dimension (inner most) respectively (29 days)
    y_train = torch.tensor(seq_data[:train_size, -1, :]).type(torch.Tensor).to(device) #  remove colon in 2nd dimension bc only interested in the last element (30th day)
    X_test = torch.tensor(seq_data[train_size:, :-1, :]).type(torch.Tensor).to(device)
    y_test = torch.tensor(seq_data[train_size:, -1, :]).type(torch.Tensor).to(device)

    model = PredictModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)

    # use Mean Squared Error as the loss — typical for regression problems.
    criterion = nn.MSELoss()

    # set up the Adam optimizer with a learning rate of 0.01
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 200

    for i in range(num_epochs):
        # call the model on training input to get predicted output.
        y_train_pred = model(X_train)

        # calculate how far off the predictions are from the actual target (y_train).
        loss = criterion(y_train_pred, y_train)

        # Prints the loss every 25 epochs to monitor progress.
        if i % 25 == 0:
            print(i, loss.item())

        optimizer.zero_grad() # reset gradients from the previous step (they accumulate otherwise).
        loss.backward() # compute the gradients via backpropagation.
        optimizer.step() # update the model's weights using the optimizer and the gradients.

    print("reached model.eval")
    model.eval()

    y_train_pred = model(X_train)
    y_test_pred = model(X_test)

    # Convert mean and std from scaler to PyTorch tensors
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    # Reshape for broadcasting (1, 1) because we are doing (batch_size, 1) * (1, 1)
    mean = mean.view(1, 1)
    std = std.view(1, 1)

    # Inverse transform: undo standardization
    y_train_pred_inv = y_train_pred * std + mean
    y_train_inv = y_train * std + mean
    y_test_pred_inv = y_test_pred * std + mean
    y_test_inv = y_test * std + mean

    # Compute RMSE
    train_rmse = manual_rmse(y_train_inv, y_train_pred_inv).item()
    test_rmse = manual_rmse(y_test_inv, y_test_pred_inv).item()

    # Align with dates
    train_dates = seq_dates[:train_size]
    test_dates = seq_dates[train_size:]

    # Flatten tensors and convert to numpy
    y_train_actual = y_train_inv.detach().cpu().numpy().flatten()
    y_train_predicted = y_train_pred_inv.detach().cpu().numpy().flatten()
    y_test_actual = y_test_inv.detach().cpu().numpy().flatten()
    y_test_predicted = y_test_pred_inv.detach().cpu().numpy().flatten()

    # Build date indexes
    train_dates = prices.iloc[:train_size+seq_length].index[-len(y_train_actual):]
    test_dates = prices.iloc[train_size+seq_length:].index[-len(y_test_actual):]

    # Build DataFrames
    train_df = pd.DataFrame({
        "Day (from 01-01-24)": train_dates,
        "Ticker": ticker,
        "Set": "Train",
        "Actual Price": y_train_actual,
        "Predicted Price": y_train_predicted,
        "Prediction Error": y_train_predicted - y_train_actual
    })

    test_df = pd.DataFrame({
        "Day (from 01-01-24)": test_dates,
        "Ticker": ticker,
        "Set": "Test",
        "Actual Price": y_test_actual,
        "Predicted Price": y_test_predicted,
        "Prediction Error": y_test_predicted - y_test_actual
    })

    # Add RMSE summary row
    metrics_df = pd.DataFrame([{
        "Ticker": ticker,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Timestamp": datetime.now()
    }])

    # Save to Excel
    train_df.to_excel(f"{ticker}_Train_Predictions.xlsx", sheet_name=f"{ticker}_Train_Predictions", index=False)
    test_df.to_excel(f"{ticker}_Test_Predictions.xlsx", sheet_name=f"{ticker}_Test_Predictions", index=False)
    metrics_df.to_excel(f"{ticker}_Metrics.xlsx", sheet_name=f"{ticker}_Metrics", index=False)
    
    print(f"Exported predictions for {ticker}.")