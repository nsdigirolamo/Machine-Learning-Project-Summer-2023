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
from tee import StdoutTee
import os
import json

dropped = ['id', 'url', 'region_url', 'image_url', 'description', 'region', 'state', 'model', 'paint_color', 'VIN', 'posting_date']
dummies = ['manufacturer', 'drive', 'fuel', 'title_status', 'transmission', 'type']
inpath = 'out\\clean\\filtered_data.csv'
outroot = 'out\\out_RF\\'
outbase = outroot + 'base\\'
outsearch = outroot + 'randomSearch\\'

def mk_heatmap(df, outpath):
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, cmap=sns.diverging_palette(100, 20, as_cmap=True))
    plt.savefig(outpath)
    plt.close()

def format_data(df, outpath, drop_cols, dummy_cols):
    df = pd.read_csv(inpath, encoding='utf8')
    #drop columns
    df.drop(drop_cols, axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df = pd.get_dummies(df, drop_first=True, columns=dummy_cols)
    df.to_csv(outpath)
    labels = df.iloc[:, df.columns != 'price'].columns
    X = df.loc[:, df.columns != 'price']
    y = df['price']
    #scale data
    X = StandardScaler().fit_transform(X)
    return X, y, labels

def search_param():
    #define gridsearch parameters
    #number of trees
    n_est = [int(x) for x in np.linspace(start=100, stop=5000, num=100)]
    #number of features to consider at each split
    #max_feat = ['sqrt', 'log2']
    max_feat = [int(x) for x in np.linspace(start=1, stop=50, num=5)]
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
    return search_parameters

def eval(model, test_features, test_labels, outpath):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    with open(outpath + 'report.txt', mode='a', encoding='utf8') as f:
        f.write('Model Performance\n')
        f.write('Average Error: {:0.4f} degrees.\n'.format(np.mean(errors)))
        f.write('Accuracy = {:0.2f}%'.format(accuracy))

def report(model, outpath, X_test, y_test):

    prediction = model.predict(X_test)
    MAE = mae(y_true=y_test, y_pred=prediction)
    accuracy = model.score(X=X_test, y=y_test)
    #accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    #feature_importance = pd.Series(model.feature_importances_, index=labels)
    #feature_importance.nlargest(25).plot(kind='barh', figsize=(10,10))

    #plt.savefig(outpath + 'feature_importance.png')
    #plt.close()
    params = model.get_params()

    with open(outpath + 'metrics.txt', mode='a', encoding='utf8') as f:
        f.write('Model Metrics:  Random Forest Regressor\n\nMean Absolute Error: ' + str(MAE) + '$.' + '\nAccuracy: ' + str(accuracy * 100) + '%.\n\nModel Parameters:')
        f.write(json.dumps(params, indent=4))
        

    with open(outpath + 'model', mode='wb') as m:
        pickle.dump(obj=model, file=m)

def plots_search(cv_results, grid_parameters):
    scores_mean = cv_results['mean_test_score']
    
###################################################################################################################################


if not os.path.exists(outbase):
    os.mkdir(outbase)

if not os.path.exists(outsearch):
    os.mkdir(outsearch)

df = pd.read_csv(inpath, encoding='utf8')
mk_heatmap(df=df, outpath=outroot + 'heatmap.png')
X, y, labels = format_data(df, outpath=outroot + 'dummy_dat.csv', drop_cols=dropped, dummy_cols=dummies)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=False)

#base model train and eval
base_model = RandomForestRegressor(n_estimators=10, random_state=40)
base_model.fit(X_train, y_train)
report(model=base_model, outpath=outbase, X_test=X_test, y_test=y_test)




#set up search cv
search_parameters = search_param()
model = RandomForestRegressor()
#change interations after testing!!!
random = RandomizedSearchCV(estimator=model, param_distributions=search_parameters, n_iter=2, cv=3, verbose=2, random_state=47, n_jobs = -1)

print('starting search')
#open log and tee output to log
#logfile = open(outsearch + 'search_log.txt', mode='w')
#original_stderr = sys.stderr
#original_stdout = sys.stdout
#sys.stdout = StdoutTee(sys.stdout, logfile,)
#sys.stderr = sys.stdout

#run hyperparameter tuning and train models
random.fit(X_train, y_train)

#reset stdout and stderr and close logfile
#sys.stdout = original_stdout
#sys.stderr = original_stdout
#logfile.close()
#print('finished search')
#select best model
best_random = random.best_estimator_

#train and fit base model


#report

report(model=best_random, outpath=outsearch, X_test=X_test, y_test=y_test)
