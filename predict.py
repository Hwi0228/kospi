import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

kospi = yf.Ticker('000660.KS')
data = kospi.history(period='1mo')
data = data.reset_index()

price = data['Close'].values.tolist()
price = [[x] for x in price]
print(price)
time = data['Date'].dt.dayofyear.values.tolist()
time = [[x] for x in time]
print(time)
regr = linear_model.LinearRegression()
regr.fit(time, price)
print(f'y={regr.coef_}x + {regr.intercept_}')
#=============== 스타일 ==================#
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
colors = cmap = plt.get_cmap('seismic')
#===========================================#

plt.scatter(time, price)
predict_time = [[x[0]+31] for x in time]
price_pred = regr.predict(predict_time)
plt.plot(predict_time, price_pred)
plt.show()