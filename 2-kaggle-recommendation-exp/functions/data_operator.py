
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

class DataOperator:
    def __init__(self, file_path):
        print("initializing DataOperator")
        self.__read_csv(file_path)
        self.__clean_df()

    def __read_csv(self, file_path):
        print("reading csv file: " + file_path)
        if file_path is None:
            raise ValueError("file_path is None")
        elif file_path.endswith(".csv") is False:
            raise ValueError("file_path is not a csv file")
        else:
            self.df = pd.read_csv(file_path)

    def __clean_df(self):
        self.df = self.df.dropna()

    def display_items(self):
        # ratingのヒストグラム
        data_rating = self.df.groupby("Rating").count()
        print(data_rating)
        plt.subplot(2, 2, 1)
        plt.bar(data_rating.index, data_rating["UserId"])
        plt.xlabel("Rating")
        plt.ylabel("Count")

        plt.subplot(2, 2, 2)
        data_eval = self.df.groupby("UserId")[["Rating"]].mean()
        print(data_eval)
        data_eval.sort_values("Rating", ascending=False, inplace=True)
        data_eval = data_eval.head(10)

        plt.xticks(rotation=90)
        plt.bar(data_eval.index, data_eval["Rating"])
        plt.show()

    # 分析するにはデータがスカスカすぎるので、次元圧縮
    def enact_svd(self):
        svd = TruncatedSVD(n_components=10)
        self.df = self.df.head(10000)
        X = self.df.pivot_table(values="Rating", index="UserId", columns="ProductId", fill_value=0)
        X = X.T
        decomposed_matrix = svd.fit_transform(X)
        print(decomposed_matrix.shape)

    
