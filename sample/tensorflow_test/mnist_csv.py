import os
import numpy as np 
import pandas as pd 

def mnist_csv_data():

    home_dir = os.path.expanduser("~")   
    train_csv_file = os.path.join(home_dir, ".keras", "mnist_train.csv")
    test_csv_file = os.path.join(home_dir, ".keras", "mnist_test.csv")

    train = pd.read_csv(train_csv_file)
    test = pd.read_csv(test_csv_file)

    x_train = train.iloc[:,1:].astype('uint8').to_numpy()
    y_train = train.iloc[:,0].astype('uint8').to_numpy()
    x_test = test.iloc[:,1:].astype('uint8').to_numpy()
    y_test = test.iloc[:,0].astype('uint8').to_numpy()

    x_train = x_train.reshape(-1,28,28)
    x_test = x_test.reshape(-1,28,28)

    return (x_train, y_train), (x_test, y_test)

