import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

csvpath = "dat/vehicles.csv"
x_cols = [
    'id', 'price', 'url', 'region_url', 'image_url', 
    'description', 'county', 'lat', 'long', 
    'posting_date'
    ]
y_cols = ['price']

df = pd.read_csv(csvpath)
df.dropna(subset=['price'], inplace=True)
X = df[x_cols].values
y = df[y_cols].values

def linear_model(in_shape, hLayers):
    model = Sequential()
    model.add(Dense(15, input_shape=in_shape, kernel_initializer='normal', activation='relu'))
    for layer in hLayers:
        model.add(Dense(layer, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model

estimator = KerasRegressor(model=linear_model(X.shape, [6,3]), epochs=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator=estimator, X=X, y=y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))



print ('hi')


