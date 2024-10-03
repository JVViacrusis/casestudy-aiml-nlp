import pandas as pd
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, data_set_path: str, label_name: str):
        self._X = None
        self._y = None
        self.df = pd.read_csv(data_set_path)
        self.label_name = label_name

    def get_X(self) -> pd.DataFrame:
        """
        Get the feature values of the dataframe. i.e. all values excluding the label values.
        :return: a dataframe with all values excluding label values
        """
        self._X = self.df.drop(self.label_name, axis=1)
        return self._X

    def get_y(self) -> pd.Series:
        """
        Get the label values of the dataframe. Returns a Series as the label values are always a vector.
        :return: a series with all values in the label column/vector
        """
        self._y = self.df[self.label_name]
        return self._y

    def get_train_test_split(
            self, train_size_percent: int, test_size_percent: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into random split for training and test data based on specified split.

        :param train_size_percent: a whole integer representing a percent for train dataset size
        :param test_size_percent: a whole integer representing a percent for test dataset size
        :return: a tuple in format of (X_train, X_test, y_train, y_test)
        """
        train_size_decimal = train_size_percent / 100
        test_size_decimal = test_size_percent / 100
        return train_test_split(
            self.get_X(), self.get_y(), train_size=train_size_decimal, test_size=test_size_decimal, random_state=42
        )
