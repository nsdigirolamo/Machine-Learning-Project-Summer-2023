import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mae, accuracy_score
from prettyprinter import pprint
import sys
import tee

dropped = ['id', 'url', 'region_url', 'image_url', 'description', 'region', 'state', 'model', 'paint_color', 'VIN', 'posting_date']
inpath = 'out\\clean\\filtered_data.csv'
outroot = 'out\\out_RF\\'

df = pd.read_csv(inpath, encoding='utf8')
#drop columns
df.drop(dropped, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
#correlation = df.corr(numeric_only=True)
#sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, cmap=sns.diverging_palette(100, 20, as_cmap=True))
#plt.savefig(outroot + 'heatmap.png')
plt.close()

df = pd.get_dummies(df, drop_first=True, columns=['manufacturer', 'drive', 'fuel', 'title_status', 'transmission', 'type'])
#df.to_csv(outroot + 'dummies.csv')

#X & y
X_h = df.iloc[:, df.columns != 'price']
labels = X_h.columns
X = df.loc[:, df.columns != 'price']
y = df['price']
#scale data
X = StandardScaler().fit_transform(X)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=False)

#define gridsearch parameters
#number of trees
n_est = [int(x) for x in np.linspace(start=100, stop=5000, num=10)]
#number of features to consider at each split
max_feat = ['auto', 'sqrt', 'log2']
#depth of each tree
max_depth = [int(x) for x in np.linspace(10, 300, num=10)]
max_depth.append(None)
#minimum sample number to split a node
min_sample_split = [2,4,5,8,10,20,50]
#minimum samples per leaf
min_sample_leaf = [1,2,4,5,8]
#sample selection method
bootstrap = [True, False]
criteria = ['absolute_error', 'squared_error']

search_parameters = {
    'n_estimators': n_est,
    'max_features': max_feat,
    'max_depth': max_depth,
    'min_samples_split': min_sample_split,
    'min_samples_leaf': min_sample_leaf,
    'bootstrap': bootstrap,
    'criterion': criteria
}
#select model, initialize random state
model = RandomForestRegressor()
random = RandomizedSearchCV(estimator=model, param_distributions=search_parameters, n_iter=100, cv=3, verbose=2, random_state=47, n_jobs = -1)


#open log and tee output to log
logfile = open(outroot + 'search_log.txt', mode='w')
original_stderr = sys.stderr
original_stdout = sys.stdout
sys.stdout = tee(sys.stdout, logfile)
sys.stderr = sys.stdout

#train
random.fit(X_train, y_train)

#reset stdout and stderr and close logfile
sys.stdout = original_stdout
sys.stderr = original_stdout
logfile.close()


#predict
prediction = model.predict(X_test)


MAE = mae(y_true=y_test, y_pred=prediction)
accuracy = model.score(X=X_test, y=y_test)
feature_importance = pd.Series(model.feature_importances_, index=X_h.columns)
feature_importance.nlargest(25).plot(kind='barh', figsize=(10,10))

plt.savefig(outroot + 'feature_importance.png')
plt.close()
#save model
with open(outroot + 'model', mode='wb') as m:
    pickle.dump(obj=model, file=m)
#save metrics
with open(outroot + 'metrics.txt', mode='a', encoding='utf8') as f:
    f.write('Model Metrics:  Random Forest Regressor\nMean Absolute Error: ' + str(MAE) + 'Accuracy: ' + str(accuracy))
