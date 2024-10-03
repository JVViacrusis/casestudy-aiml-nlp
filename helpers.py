import pandas as pd
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, data_set_path, label_name):
        self._X = None
        self._y = None
        self.df = pd.read_csv(data_set_path)
        self.label_name = label_name

    def get_X(self):
        self._X = self.df.drop(self.label_name, axis=1)
        return self._X

    def get_y(self):
        self._y = self.df[self.label_name]
        return self._y

    def get_train_test_split(self, train_size_percent, test_size_percent):
        train_size_decimal = train_size_percent / 100
        test_size_decimal = test_size_percent / 100
        return train_test_split(
            self.get_X(), self.get_y(), train_size=train_size_decimal, test_size=test_size_decimal, random_state=42
        )
