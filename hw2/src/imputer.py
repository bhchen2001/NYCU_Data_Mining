from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
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

        # print("=====================================")
        # print("=          Imputation               =")
        # print("=       using Simple Imputer        =")
        # print("=====================================")
        # categorical_feature = ['hospital_id'
        #                         , 'elective_surgery'
        #                         , 'ethnicity'
        #                         , 'gender'
        #                         , 'icu_admit_source'
        #                         , 'icu_id'
        #                         , 'icu_stay_type'
        #                         , 'icu_type'
        #                         , 'apache_post_operative'
        #                         , 'arf_apache'
        #                         , 'gcs_unable_apache'
        #                         , 'intubated_apache'
        #                         , 'ventilated_apache'
        #                         , 'aids'
        #                         , 'cirrhosis'
        #                         , 'diabetes_mellitus'
        #                         , 'hepatic_failure'
        #                         , 'immunosuppression'
        #                         , 'leukemia'
        #                         , 'lymphoma'
        #                         , 'solid_tumor_with_metastasis'
        #                         , 'apache_3j_bodysystem'
        #                         , 'apache_2_bodysystem']
        # imputer = SimpleImputer(strategy='most_frequent')
        # imputed_data = imputer.fit_transform(self.merged_df[categorical_feature])
        # self.merged_df[categorical_feature] = pd.DataFrame(imputed_data, columns=categorical_feature)
        # imputer = SimpleImputer(strategy='median')
        # non_category_feature = list(set(self.merged_df.columns) - set(categorical_feature))
        # imputed_data = imputer.fit_transform(self.merged_df[non_category_feature])
        # self.merged_df[non_category_feature] = pd.DataFrame(imputed_data, columns=non_category_feature)

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