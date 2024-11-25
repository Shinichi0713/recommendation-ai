
import pandas as pd


class DataOperator:
    def __init__(self, file_path):
        print("initializing DataOperator")
        self.__read_csv(file_path)

    def __read_csv(self, file_path):
        print("reading csv file: " + file_path)
        if file_path is None:
            raise ValueError("file_path is None")
        elif file_path.endswith(".csv") is False:
            raise ValueError("file_path is not a csv file")
        else:
            self.df = pd.read_csv(file_path)

