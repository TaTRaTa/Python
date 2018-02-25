import datetime
import pickle
import quandl
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

np.set_printoptions(linewidth=255)
style.use('ggplot')
data = quandl.get('WIKI/GOOGL')

data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume', 'Adj. Close']]
data['HL_PCT'] = (data['Adj. High'] - data['Adj. Open']) / data['Adj. Open']* 100
data['PCT_Change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100

forecast_col = 'Adj. Close'
data.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

print('------------------------------------------------------------------------------------')
x = np.array(data.drop('label', 1))
# x = preprocessing.normalize(x)
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# # Lines below must be executed only first run time
# clr = LinearRegression()
# # clr = svm.SVR(kernel='poly')
# clr.fit(x_train, y_train)
# # lines below must be executed only first run time
# with open('linear_reg.pickle', 'wb') as f:
#     pickle.dump(clr, f)

pickle_in = open('linear_reg.pickle', 'rb')
clr = pickle.load(pickle_in)
accuracy = clr.score(x_test, y_test)

# print(accuracy)
forecast_set = clr.predict(x_lately)
print(forecast_set, accuracy, forecast_out)

data['Forecast'] = np.nan
last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns) - 1)] + [i]

data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print('------------------------------------------------------------------------------------')
# print(data[-36:-34])