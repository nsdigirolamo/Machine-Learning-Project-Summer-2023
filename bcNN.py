import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

csvpath = "dat/dummy_dat.csv"
#csvpath = "dat/housing.csv"
#csvpath = "dat/vehicles.csv"



y_cols = ['price']

df = pd.read_csv(csvpath)
df.dropna(inplace=True)
labels = df.columns
X = df.loc[:, ~df.columns.isin(y_cols)].values
y = df[y_cols].values

#df = pd.read_csv(csvpath, header=None, delim_whitespace=True)

def model():
    model = Sequential()
#test at 13 for houseing dataset, 
    model.add(Dense(X.shape[1], input_shape=(X.shape[1],), kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model

estimator = KerasRegressor(model=model(), epochs=100, batch_size=50, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator=estimator, X=X, y=y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


