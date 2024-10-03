from helpers import DataSet

dataset = DataSet("IMDB Dataset.csv", "sentiment")

print("x", dataset.get_X())
print("y", dataset.get_y())

train_X, test_X, train_y, test_y = dataset.get_train_test_split(60, 40)

print("train x length", len(train_X))
print("test x length", len(test_X))
