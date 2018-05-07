import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = load_boston()
X = pd.DataFrame(data.data)
X.columns = data.feature_names
y = data.target
min_max_scaler = preprocessing.MinMaxScaler()
y = y.reshape((y.shape[0], 1))
X = min_max_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=5)

