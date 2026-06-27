import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

kospi = yf.Ticker('GC=F')
data = kospi.history(start="2025-06-27",end="2026-06-27")
data = data.reset_index()

price = data['Close'].values.tolist()
price = [[x] for x in price]

time_day = data['Date'].dt.dayofyear.values.tolist()
time_year = data['Date'].dt.year.values.tolist()
time = [[x[0]+365*x[1]] for x in zip(time_day,time_year)]

poly_feature = preprocessing.PolynomialFeatures(degree=5, include_bias=False)
time_poly = poly_feature.fit_transform(time)
print('A_poly =', time_poly)

regr = linear_model.LinearRegression()
regr.fit(time_poly, price)
print(f'y={regr.coef_}x + {regr.intercept_}')
#=============== 스타일 ==================#
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
colors = cmap = plt.get_cmap('seismic')
#===========================================#

plt.scatter(time, price)

predict_time = time.copy()

last_day = time[-1][0]
future_time = [[last_day + i] for i in range(1, 365)]
predict_time.extend(future_time)

predict_time_poly = poly_feature.fit_transform(predict_time)
y_pred = regr.predict(predict_time_poly)

plt.plot(predict_time, y_pred, 'r')
plt.show()