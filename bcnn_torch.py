
import torch.nn as nn
import torch as tf
import torch.optim as optim

import pandas as pd
import matplotlib.pylab as plt
#import seaborn as sns
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError
#import pickle
import warnings


dropped = ['id', 'url', 'region_url', 'image_url', 'description', 'region', 'model', 'paint_color', 'VIN', 'posting_date', 'lat', 'long']
dummies = ['manufacturer', 'drive', 'fuel', 'title_status', 'transmission', 'type', 'state']

inpath = 'data\\filtered_data.csv'

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Linear(128, output_dim)           
        )
    def forward(self, x):
        return self.layers(x)


class Dataset(Dataset):
    def __init__(self, X, y):
        self.y = y
        self.X = X
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        label = self.y[idx]
        data = self.X[idx]
        #sample = {'Data': data, 'Label': label}
        return data, label
    

if __name__ == '__main__':
    #data bullshit
    ################################################################
    df = pd.read_csv(inpath, encoding='utf8')
    #drop columns
    df.drop(dropped, axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df = pd.get_dummies(df, drop_first=True, columns=dummies)
    features = list(df.iloc[:, df.columns != 'price'].columns)

    ct = ColumnTransformer(
        [
            ('scaler', StandardScaler(), [0, 1]), 
            ('passthrough', 'passthrough', slice(2, None))
        ],
        )
    
    

    features = df.iloc[:, df.columns != 'price'].columns
    X = df.loc[:, df.columns != 'price'].values
    y = df['price'].values
    
    ct.fit(X)
    X = pd.DataFrame(ct.transform(X))
    #scale data
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)
   


    #X_train_scaled = ct.transform(X_train)
    #X_test_scaled = ct.transform(X_test)
    X_train_arr = np.asarray(X_train).astype('float32')
    X_test_arr = np.asarray(X_test).astype('float32')
    y_train_arr = np.asanyarray(y_train).astype('float32')
    y_test_arr = np.asanyarray(y_test).astype('float32')

    #TD_train = Dataset(X_test_scaled_arr, y_train_arr)
    #TD_test = Dataset(X_test_scaled_arr, y_test_arr)


    #end bullshit
    ###################################################################







    device = tf.device('cuda' if tf.cuda.is_available() else 'cpu')
    input_dim = X_train_arr.shape[1]
    output_dim = 1

    model = LinearRegressionModel(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=0.0)
    criterion = nn.MSELoss()
    batch_size = 1000

    X_train_tensor = tf.from_numpy(X_train_arr)
    X_test_tensor = tf.from_numpy(X_test_arr)
    y_train_tensor = tf.from_numpy(y_train_arr).reshape(-1, 1)
    y_test_tensor = tf.from_numpy(y_train_arr).reshape(-1, 1)

    train_dataset = Dataset(X=X_train_tensor, y=y_train_tensor)
    test_dataset = Dataset(X=X_test_tensor, y=y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=12)
    mae = MeanAbsoluteError().to(device)

    G_epochs = []
    G_loss = []
    G_mae = []
    for epoch in range(1000):
        epoch += 1
        loss = 0.0
        #inputs = torch.from_numpy(X_train_scaled_arr).to(device, torch.float32)
        #labels = torch.from_numpy(y_train_arr).to(device, torch.float32)
        for batch_idx, (inputs, targets) in enumerate(train_dataloader,0):
            warnings.simplefilter('ignore')
            #inputs, labels = data

            #inputs, targets = inputs.to(device, torch.float32), 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                error = mae(outputs, targets)
        print('Epoch: {} |  Train Loss: {} | MSE: {}'.format(epoch, loss, error))
        G_epochs.append(epoch)
        G_loss.append(loss)
        G_mae.append(error)
    with open('src\\NN_model', mode='wb') as f:
        model.to('cpu')
        pickle.dump(model, f)

    stats = list(zip(G_epochs, G_loss, G_mae))

    with open('src\\stats', mode='wb') as f:
        pickle.dump(stats, f)


