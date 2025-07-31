import numpy as np # for working with arrays
import pandas as pd # for working with data set, data frame structure
import yfinance as yf # for using Yahoo finance API
import matplotlib.pyplot as plt # for data visualization

test_df = pd.read_excel("AAPL_Test_Predictions.xlsx")
train_df = pd.read_excel("AAPL_Train_Predictions.xlsx")
metrics_df = pd.read_excel("AAPL_Metrics.xlsx")

# visualize and plot the following:
# 1. 3/4 of the plot with a price and the prediction
# 2. below that visualize the error
# 3. for each point and time how large the error is
# 4. the rmse

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(4, 1) # 4 rows, 1 column

# Plot actual vs predicted prices
axis1 = fig.add_subplot(gs[:3, 0]) # fill the first 3 rows and the first column
axis1.plot(test_df['Day (from 01-01-24)'], test_df['Actual Price'], color='blue', label='Actual Price')
axis1.plot(test_df['Day (from 01-01-24)'], test_df['Predicted Price'], color='green', label='Predicted Price')
axis1.legend()
axis1.set_title('AAPL Stock Price Prediction')
axis1.set_xlabel('Date')
axis1.set_ylabel('Price')

# Plot prediction error
axis2 = fig.add_subplot(gs[3, 0])
axis2.axhline(metrics_df['Test RMSE'].iloc[0], color='blue', linestyle='--', label='RMSE')
axis2.plot(test_df['Day (from 01-01-24)'], test_df['Prediction Error'], color='red', label='Prediction Error')
axis2.legend()
axis2.set_title('Prediction Error')
axis2.set_xlabel('Date')
axis2.set_ylabel('Error')

plt.tight_layout()
plt.show()