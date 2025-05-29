
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns




# Load the dataset
file_path = r"C:\Users\hp\Desktop\py\stock predictor\Netflix Stock Market.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Convert date column to datetime and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Calculate technical indicators
df['SMA'] = df['Close'].rolling(window=20).mean()
df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

# Calculate MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
# df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculate Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
df['BB_Upper'] = df['BB_Middle'] + 2*df['Close'].rolling(window=20).std()
df['BB_Lower'] = df['BB_Middle'] - 2*df['Close'].rolling(window=20).std()

# df.to_csv("C:/Users/hp/Desktop/py/stock predictor/netflix1.csv")


# Drop rows with NaN values created by rolling calculations
df.dropna(inplace=True)


#Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD) and Bollinger Bands


# Select the relevant columns for prediction
features = ['Close', 'SMA', 'EMA', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower']

data = df[features].values

plt.figure(figsize=(12, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Technical Indicators')
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the dataset for RNN
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step),1:])
        Y.append(data[i + time_step, 0])  # Analyse the 'Close' price
    return np.array(X), np.array(Y)

time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Split data into training and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True)),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(15),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.summary()

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

# Predict and inverse the normalization
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

# Prepare for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, 0] = train_predict

testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
test_start_index = len(train_predict) + (time_step)

if test_start_index + len(test_predict) <= len(testPredictPlot):
    testPredictPlot[test_start_index:test_start_index + len(test_predict), 0] = test_predict
else:
    testPredictPlot[test_start_index:, 0] = test_predict[:len(testPredictPlot) - test_start_index, 0]

# Plot the results
plt.figure(figsize=(16, 8))
dates = df['Date']

# plt.plot(dates, scaler.inverse_transform(scaled_data)[:, 0], label='Actual Stock Price')
plt.plot(dates[time_step:len(train_predict) + time_step], trainPredictPlot[time_step:len(train_predict) + time_step, 0], label='Train Prediction')
plt.plot(dates[test_start_index:test_start_index + len(test_predict)], testPredictPlot[test_start_index:test_start_index + len(test_predict), 0], label='Test Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






print('errors',model.evaluate(X_train,y_train))
print('errors',model.evaluate(X_test,y_test))