from t_sne import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def test_t_SNE():
    tsne = TSNE(100, 10, 0.5)

    X = pd.read_csv("data/mnist_train.csv")
    # X["img"] = X.iloc[:, 1:].values.tolist()
    # X = X.loc[:, ["label", "img"]]

    def select_digits(x):
        digits = [0, 1, 3]
        if x.name not in digits:
            return None
        return x.iloc[:100, :]

    X_selected = X.groupby(by=["label"]).apply(select_digits)
    X_selected = X_selected.reset_index(drop=True)

    X_numpy = X_selected.iloc[:, 1:].to_numpy()

    print(X_numpy.shape)
    print(X_numpy[:5])
    print(X_numpy.max(), X_numpy.min())

    Y = tsne.fit(X)

    with open("data/reduced.npy", "wb") as f:
        np.save(f, Y)

    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


test_t_SNE()
