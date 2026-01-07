import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

# ---------------------------
# 1. Télécharger les données
# ---------------------------

TICKERS = ["VK.PA", "APAM.AS", "AIR.PA"]  # Vallourec, Aperam, Airbus
data = yf.download(TICKERS, start="2018-01-01", end="2025-10-01")["Close"]

# Sélection d’un titre pour le MVP (on commence simple)
df = data["VK.PA"].dropna().to_frame()
df.columns = ["Price"]

# ---------------------------
# 2. Prétraitement
# ---------------------------

scaler = MinMaxScaler()
df["Scaled"] = scaler.fit_transform(df)

SEQ_LEN = 30

def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(df["Scaled"].values)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape((-1, SEQ_LEN, 1))
X_test = X_test.reshape((-1, SEQ_LEN, 1))

# ---------------------------
# 3. Modèle baseline LSTM
# ---------------------------

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# ---------------------------
# 4. Évaluation
# ---------------------------

pred = model.predict(X_test)
pred_rescaled = scaler.inverse_transform(pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(y_test_rescaled, pred_rescaled))
mape = mean_absolute_percentage_error(y_test_rescaled, pred_rescaled)

print("RMSE:", rmse)
print("MAPE:", mape)

# ---------------------------
# 5. Graphique
# ---------------------------

plt.figure(figsize=(12,5))
plt.plot(y_test_rescaled, label="Réalité")
plt.plot(pred_rescaled, label="Prédiction (baseline LSTM)")
plt.title("MVP — Baseline LSTM sur VK.PA")
plt.legend()
plt.show()