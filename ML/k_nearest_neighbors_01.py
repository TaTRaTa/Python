import numpy as np
import pandas as pd
from sklearn import cross_validation, neighbors


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop('id', 1, inplace=True)

X = df.drop('Class', 1)
y = df['Class']

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clr = neighbors.KNeighborsClassifier()
clr.fit(x_train, y_train)

accuracy = clr.score(x_test, y_test)
print(accuracy)

measures = np.array([4,2,1,1,1,2,3,2,1])
measures = measures.reshape(1, -1)
predict = clr.predict(measures)
print(predict)

