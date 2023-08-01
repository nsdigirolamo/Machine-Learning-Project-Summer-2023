import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, f1_score as f1

dropped = ['id', 'url', 'region_url', 'image_url', 'description', 'region', 'state', 'model', 'paint_color', 'VIN', 'posting_date']
inpath = 'out\\clean\\filtered_data.csv'
outroot = 'out\\out_RF\\'

df = pd.read_csv(inpath, encoding='utf8')
#drop columns
df.drop(dropped, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, cmap=sns.diverging_palette(100, 20, as_cmap=True))
plt.savefig(outroot + 'heatmap.png')
plt.close()

df = pd.get_dummies(df, drop_first=True, columns=['manufacturer', 'drive', 'fuel', 'title_status', 'transmission', 'type'])
df.to_csv(outroot + 'dummies.csv')

#X & y
X_h = df.iloc[:, df.columns != 'price']
labels = X_h.columns
X = df.loc[:, df.columns != 'price']
y = df['price']
#scale data
X = StandardScaler().fit_transform(X)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=False)
#select model, initialize random state
model = RandomForestRegressor(random_state=True)
#train
model.fit(X_train, y_train)
#predict
prediction = model.predict(X_test)

MAE = mae(y_true=y_test, y_pred=prediction)
#F1 = f1(y_true=y_test, y_pred=prediction, labels=labels)
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
