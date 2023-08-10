
import torch.nn as nn
import torch as tf
import torch.optim as optim

import pandas as pd
import matplotlib.pylab as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError



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
    y_test_tensor = tf.from_numpy(y_test_arr).reshape(-1, 1)

    train_dataset = Dataset(X=X_train_tensor, y=y_train_tensor)
    test_dataset = Dataset(X=X_test_tensor, y=y_test_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=12)
    mae = MeanAbsoluteError().to(device)


    
    best_err = 10000000000000000000
    Train_epochs = []
    Train_loss = []
    Train_mae = []
    Test_epochs = []
    Test_loss = []
    Test_mae = []
    for epoch in range(10):
        epoch += 1
        train_loss = 0.0
        test_loss = 0.0
        error = []
        loss = []
        #######################
        #training
        ######################
        for batch_idx, (inputs, targets) in enumerate(train_dataloader,0):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            t_loss = criterion(outputs, targets)
            loss.append(t_loss.item())
            t_error = mae(outputs, targets)
            error.append(t_error.item())
            t_loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print('Epoch: {}, Batch: {} |  Train Loss: {} | MAE: {}'.format(epoch, batch_idx, t_loss, t_error))
        avg_train_error = sum(error) / len(error)   
        avg_train_loss = sum(loss) / len(loss) 
        Train_epochs.append(epoch)
        Train_loss.append(avg_train_loss)
        Train_mae.append(avg_train_error)
        ######################
        #testing
        ######################
        error = []
        loss = []
        with tf.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_dataloader,0):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                t_loss = criterion(outputs, targets)
                loss.append(t_loss.item())
                error.append(mae(outputs, targets).item())
            avg_test_loss = sum(loss) / len(loss)
            avg_test_error = sum(error) / len(error)
            print('**TEST** | Epoch: {} |  Average Test Loss: {} | Average MAE: {}\n\n'.format(epoch, avg_test_loss, avg_test_error))
            Test_epochs.append(epoch)
            Test_loss.append(avg_test_loss)
            Test_mae.append(avg_test_error)
            if avg_test_error < best_err:
                print('**New High Score**')
                print('saving state...')
                best_err = avg_test_error
                best_model_stats = {
                    'state': model.state_dict(),
                    'MAE': avg_test_error,
                    'epoch': 0.0
                }
                tf.save(best_model_stats, 'C:\\Users\\bcox\\Desktop\\hw3\\src\\nn_out\\checkpoint.pth')
                print('serializing model')
                with open('C:\\Users\\bcox\\Desktop\\hw3\\src\\nn_out\\NN_model.pkl', mode='wb') as f:
                    model.to('cpu')
                    pickle.dump(model, f)
                    model.to(device)
                print('checkpoint complete\n\n')


    

    train_stats = list(zip(Train_epochs, Train_loss, Train_mae))
    test_stats = list(zip(Test_epochs, Test_loss, Test_mae))

    with open('C:\\Users\\bcox\\Desktop\\hw3\\src\\nn_out\\train_stats.pkl', mode='wb') as f:
        pickle.dump(train_stats, f)

    with open('C:\\Users\\bcox\\Desktop\\hw3\\src\\nn_out\\test_stats.pkl', mode='wb') as f:
        pickle.dump(test_stats, f)


