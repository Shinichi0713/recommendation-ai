

import os, sys
import functions


def main():
    print("executing main.py")
    dir_current = os.path.dirname(os.path.abspath(__file__))
    data_operator = functions.DataOperator(f"{dir_current}/data/ratings_Beauty.csv")
    data_operator.enact_svd()
    print(data_operator.recommend_items('0733001998'))


if __name__ == '__main__':
    print("executing main.py")
    main()

