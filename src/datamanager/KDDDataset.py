import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, Sampler
from torch.utils.data.dataset import T_co
import pandas as pd
import numpy as np
from sklearn import preprocessing


class KDDDataset(Dataset):
    """
    This class is used to load KDD Cup dataset as a pytorch Dataset
    """

    def __init__(self, path='../data/kddcup_data'):
        self.path = path
        data = self._load_data(path)
        self.X, self.y = data[:, :-1], data[:, -1]

        # Normalize data
        # X_mean = np.mean(self.X, axis=1)
        # X_std = np.std(self.X, axis=1)
        # self.X = (self.X - X_mean)/X_std

        scaler = preprocessing.MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.n = len(data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def get_data_index_by_label(self, label):
        indices = np.array([i for i in range(self.n)])
        return indices[self.y == label]

    def _load_data(self, path):
        LABEL_INDEX = 41
        df = pd.read_csv(path, header=None)

        # regroup all the abnormal data to attack
        df[LABEL_INDEX] = df[LABEL_INDEX].map(lambda x: 1 if x == 'normal.' else 0)

        # transform categorical variables to one hot
        df_cat_one_hot = pd.get_dummies(df)
        df_cat_one_hot.drop(columns=[LABEL_INDEX], inplace=True)
        df_cat_one_hot['label'] = df[LABEL_INDEX]
        # normal_data = np.array(df_cat_one_hot[df_cat_one_hot[41] == 1])

        return np.array(df_cat_one_hot, dtype="float32")

    def split_train_test(self, test_perc=.2, seed=0):
        torch.manual_seed(seed)

        num_test_sample = int(self.n * test_perc)
        shuffled_idx = torch.randperm(self.n).long()
        train_set = Subset(self, shuffled_idx[num_test_sample:])
        test_set = Subset(self, shuffled_idx[:num_test_sample])

        return train_set, test_set

    def get_shape(self):
        return self.X.shape
