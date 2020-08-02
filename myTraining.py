import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from . import __check_build
import pickle

def data_split(data,ratio):  # ratio = % of training or testing data
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))  # It will pick random values from data frame
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]# Those rows which are my test data rows, takes all rows and colums from test_set_size
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

    if __name__ == "__main__":

        #Reading the data

        df = pd.read_csv('Data.csv')
        train,test = data_split(df,0.2)
        X_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
        X_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
        Y_train = train[['infectionProb']].to_numpy().reshape(2000,)
        Y_test = test[['infectionProb']].to_numpy().reshape(499,)
        clf = LogisticRegression()
        clf.fit(X_train,Y_train)
        clf = LogisticRegression(solver='lbfgs', multi_class='auto')
        clf.fit(X_train,Y_train)

        # open a file, where you ant to store the data
        file = open('model.pkl', 'wb')

        # dump information to that file
        pickle.dump(clf, file)
        file.close()

     