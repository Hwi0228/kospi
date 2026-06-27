import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

kospi = yf.Ticker('^KS11')
data = kospi.history(start="1996-12-01",end="2026-06-27")
data = data.reset_index()

price = data['Close'].values.tolist()
price = [[x] for x in price]

time_day = data['Date'].dt.dayofyear.values.tolist()
time_year = data['Date'].dt.year.values.tolist()
time = [[x[0]+365*x[1] - 700000] for x in zip(time_day,time_year)]

scaler_X = preprocessing.StandardScaler()
scaler_y = preprocessing.StandardScaler()

time_scaled = scaler_X.fit_transform(time)
price_scaled = scaler_y.fit_transform(price)

regr = SVR(kernel='rbf', C=5, epsilon=0)
regr.fit(time_scaled, price_scaled.ravel()) 

#=============== 스타일 ==================#
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
colors = cmap = plt.get_cmap('seismic')
#===========================================#

plt.scatter(time, price)

predict_time = time.copy()

last_day = time[-1][0]
future_time = [[last_day + i] for i in range(1, 3650)]
predict_time.extend(future_time)

predict_time_scaled = scaler_X.transform(predict_time)
y_pred_scaled = regr.predict(predict_time_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

plt.plot(predict_time, y_pred, 'r')
plt.show()