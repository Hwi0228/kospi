import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime

kospi =  yf.Ticker('^KS11')
data = kospi.history(period='1mo')
data = data.reset_index()
print(data)
price = data['Close'].values.tolist()
price = [x for x in price]

time = data['Date'].dt.dayofyear.values.tolist()
time = [[x] for x in price]

regr = linear_model.LinearRegression()
regr.fit(time, price)

#=============== 스타일 ==================#
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
colors = cmap = plt.get_cmap('seismic')
#===========================================#

plt.scatter(time, price)
price_pred = regr.predict(price)
plt.plot(time, price_pred)
plt.show()