import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

from sklearn.datasets import fetch_openml
data = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
X, y = shuffle(data.data, data.target, random_state=7)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=10)
sv_regressor.fit(X_train, y_train)

y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
test_data1 = [3.5, 0, 18.1, 1, 0.86, 6.0, 90, 2.51, 24, 666, 20.2, 350.0, 15.0]
test_data2 = [0.05, 40, 5.0, 1, 0.45, 7.5, 30, 5.5, 4, 250, 16.0, 390.0, 4.5]
test_data3 = [10.5, 0, 25.0, 0, 0.75, 5.0, 98, 1.8, 24, 700, 21.0, 300.0, 25.0]

print("\nPredicted price:", sv_regressor.predict([test_data])[0])

print("\nPredicted price1:", sv_regressor.predict([test_data1])[0])
print("\nPredicted price2:", sv_regressor.predict([test_data2])[0])
print("\nPredicted price3:", sv_regressor.predict([test_data3])[0])
