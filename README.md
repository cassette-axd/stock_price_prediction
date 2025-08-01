# Stock Market Prediction

## NOTE
This AI model for this project was made for learning purposes only and should not be used for making actual investments or means for making profit as the model isn't fully accurate and is built on a small scale.

## Objective
To build a real-time stock market prediction system using a streaming data pipeline. The project leverages an LSTM-based regression model to forecast future stock prices, with results streamed via Kafka and visualized for trend analysis. This tool is designed to support traders, analysts, and data enthusiasts in understanding short-term price movements and improving decision-making with predictive insights.

## Challenges
Developing this pipeline involved several technical and analytical challenges:

- Processing sequential stock price data while maintaining temporal order for time-series forecasting.

- Scaling and normalizing financial data to improve model performance.

- Integrating real-time data flow between producer and consumer applications using Kafka.

- Managing Dockerized services for reproducibility and ease of deployment.

- Ensuring accurate model evaluation through train-test splitting and error measurement.

## Solution
Implemented a streaming pipeline where:

- Producer collects historical stock price data (via Yahoo Finance API) and streams it into Kafka.

- Consumer receives the data, trains an LSTM model in PyTorch, and predicts future stock prices.

- Predictions are inverse-transformed to original price values and compared against actual prices.

- Results, including prediction errors, are exported to Excel and visualized for clearer trend interpretation.

This setup enables a continuous loop of prediction, evaluation, and visualization in near real-time.

## Tools and Technologies
- Python: Data preprocessing, model training, and evaluation

- PyTorch: LSTM-based regression model for time-series forecasting

- Apache Kafka: Real-time message streaming between producer and consumer

- Docker: Containerized pipeline for reliable deployment

- Excel: Storing and exporting prediction results

- Matplotlib: Visualizing actual vs predicted stock prices and errors

## Features
- Real-time streaming of stock price data from producer to consumer

- LSTM neural network predictions of future prices

- Export of results (predicted vs actual prices and errors) into Excel

- Visual dashboards showing price trends and prediction accuracy

- Dockerized setup for easy reproducibility and scaling

## How to Run Project
Make sure you have the following installed:

- Docker Desktop

- Python 3.12+

- Required Python packages (install via pip install -r requirements.txt)

### Step 1: Start Kafka with Docker
- docker-compose pull
  
- docker-compose up -d

Check if containers are running:
- docker ps

### Step 2: Run the Producer
Open a terminal and run:
- python producer.py

### Step 3: Run the Consumer
Open a new terminal and run:
- python consumer.py
The consumer receives data, trains the LSTM model, makes predictions, and saves results as excel sheets.

### Step 4: View the results
Once the consumer saves the predictions as excel sheets, run in a terminal:
- python plot.py
