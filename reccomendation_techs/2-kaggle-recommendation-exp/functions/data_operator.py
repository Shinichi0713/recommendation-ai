
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        data_eval.sort_values("Rating", ascending=False, inplace=True)
        data_eval = data_eval.head(10)

        plt.xticks(rotation=90)
        plt.bar(data_eval.index, data_eval["Rating"])
        plt.show()

    # 分析するにはデータがスカスカすぎるので、次元圧縮
    def enact_svd(self):
        # data整形
        self.df = self.df.head(10000)
        self.X = self.df.pivot_table(values="Rating", index="UserId", columns="ProductId", fill_value=0)
        self.X = self.X.T
        # 次元圧縮
        svd = TruncatedSVD(n_components=10)
        decomposed_matrix = svd.fit_transform(self.X)

        # 相関行列化(これが推薦の元になる)
        self.correlation_recommendation = np.corrcoef(decomposed_matrix)

    def recommend_items(self, item_id, num=10):
        if self.correlation_recommendation is None:
            raise ValueError("correlation_recommendation is None")
        else:
            product_names = list(self.X.index)
            product_ID = product_names.index(item_id)
            correlation_product_ID = self.correlation_recommendation[product_ID]
            recommend = list(self.X.index[correlation_product_ID > 0.9])
            # ユーザーが購入した商品を除外
            recommend.remove(item_id)
            return recommend[0:num]
