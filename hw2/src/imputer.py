from sklearn.impute import KNNImputer
import pandas as pd

class Imputer():
    def __init__(self, train_data, test_data, k = 5):
        self.train_data = train_data
        self.test_data = test_data
        self.merged_df = None
        self.k = k

    def merge_data(self):
        # merge train and test data
        self.merged_df = pd.concat([self.train_data, self.test_data])

    def impute(self):
        print("=====================================")
        print("=          Imputation               =")
        print("=       using KNN Imputer           =")
        print("=====================================")
        imputer = KNNImputer(n_neighbors=self.k)
        imputed_data = imputer.fit_transform(self.merged_df)
        self.merged_df = pd.DataFrame(imputed_data, columns=self.merged_df.columns)

        # split train and test data
        self.train_data = self.merged_df.iloc[:len(self.train_data), :]
        self.test_data = self.merged_df.iloc[len(self.train_data):, :]

    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
    
    def process(self):
        self.merge_data()
        print("Imputation Started")
        self.impute()
        print("Imputation Completed")