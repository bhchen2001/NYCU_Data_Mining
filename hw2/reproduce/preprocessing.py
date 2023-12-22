import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imputer import Imputer

class Preprocessing():
    def __init__(self, train_data, train_label, test_data):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data

    def drop_useless(self):
        # drop the useless columns
        useless_features = ['encounter_id', 'patient_id']
        self.train_data = self.train_data.drop(useless_features, axis=1)
        self.test_data = self.test_data.drop(useless_features, axis=1)

    def categorical_processing(self):
        categorical_feature = ['hospital_id'
                                , 'elective_surgery'
                                , 'ethnicity'
                                , 'gender'
                                , 'icu_admit_source'
                                , 'icu_id'
                                , 'icu_stay_type'
                                , 'icu_type'
                                , 'apache_post_operative'
                                , 'arf_apache'
                                , 'gcs_unable_apache'
                                , 'intubated_apache'
                                , 'ventilated_apache'
                                , 'aids'
                                , 'cirrhosis'
                                , 'diabetes_mellitus'
                                , 'hepatic_failure'
                                , 'immunosuppression'
                                , 'leukemia'
                                , 'lymphoma'
                                , 'solid_tumor_with_metastasis'
                                , 'apache_3j_bodysystem'
                                , 'apache_2_bodysystem']
        binary_feature = ['elective_surgery', 
                          'gender', 
                          'apache_post_operative', 
                          'arf_apache', 
                          'gcs_unable_apache', 
                          'intubated_apache', 
                          'ventilated_apache', 
                          'aids', 
                          'cirrhosis', 
                          'diabetes_mellitus', 
                          'hepatic_failure', 
                          'immunosuppression', 
                          'leukemia', 
                          'lymphoma', 
                          'solid_tumor_with_metastasis']
        
        # dealing with outliers
        # fill the missing values of gender with 't'
        self.train_data['gender'].fillna('t', inplace=True)

        # turn the unreasonable values of feature contains 'prob' to nan
        prob_feature = self.train_data.columns[self.train_data.columns.str.contains('prob')]
        for feature in prob_feature:
            self.train_data[feature] = self.train_data[feature].apply(lambda x: x if x <= 1 and x > 0 else None)
            self.test_data[feature] = self.test_data[feature].apply(lambda x: x if x <= 1 and x > 0 else None)

        # option1: drop the categorical features

        # print("=====================================")
        # print("drop non-binary categorical features")
        # print("=====================================")

        # replace male to 0 and female to 1
        # self.train_data['gender'].replace({'M': 0, 'F': 1}, inplace=True)
        # self.test_data['gender'].replace({'M': 0, 'F': 1}, inplace=True)

        # drop non-binary categorical features
        # non_binary_feature = list(set(categorical_feature) - set(binary_feature))
        # self.train_data = self.train_data.drop(non_binary_feature, axis=1)
        # self.test_data = self.test_data.drop(non_binary_feature, axis=1)

        # option2: use one-hot encoding
        # self.train_data = pd.get_dummies(self.train_data, columns=categorical_feature)
        # self.test_data = pd.get_dummies(self.test_data, columns=categorical_feature)

        # option3: use frequency encoding
        # consider both train and test data
        # print("=====================================")
        # print("=          Frequency Encoding       =")
        # # print("Only Non-binary categorical features")
        # print("=====================================")
        # # non_binary_feature = list(set(categorical_feature) - set(binary_feature))
        # for feature in categorical_feature:
        #     freq = pd.concat([self.train_data[feature], self.test_data[feature]]).value_counts()
        #     self.train_data[feature] = self.train_data[feature].map(freq)
        #     self.test_data[feature] = self.test_data[feature].map(freq)

        # option4: combining frequency encoding and label encoding
        # count the died sum for each category
        # print("=====================================")
        # print("=     Frequency / Label Encoding    =")
        # print("Only Non-binary categorical features")
        # print("=====================================")
        # labeled_train_data = pd.concat([self.train_data, self.train_label], axis=1)
        # non_binary_feature = list(set(categorical_feature) - set(binary_feature))
        # for feature in non_binary_feature:
        #     freq = labeled_train_data.groupby(feature)['has_died'].sum()
        #     self.train_data[feature] = self.train_data[feature].map(freq)
        #     self.test_data[feature] = self.test_data[feature].map(freq)

        # option5: label encoding
        print("=====================================")
        print("=            Label Encoding         =")
        print("Only Non-binary categorical features")
        print("=====================================")
        labeled_train_data = pd.concat([self.train_data, self.train_label], axis=1)
        non_binary_feature = list(set(categorical_feature) - set(binary_feature))
        for feature in categorical_feature:
            mean = labeled_train_data.groupby(feature)['has_died'].mean()
            self.train_data[feature] = self.train_data[feature].map(mean)
            self.test_data[feature] = self.test_data[feature].map(mean)


    def data_details(self):
        # show the details and information of dataset
        print("=====================================")
        print("train data information:")
        print(self.train_data.info())
        print(self.train_data.describe())
        print(self.train_data.isnull().sum())
        print(self.train_data.head())
        print("=====================================")

        print("=====================================")
        print("test data information:")
        print(self.test_data.info())
        print(self.test_data.describe())
        print(self.test_data.isnull().sum())
        print(self.test_data.head(3))
        print("=====================================")

    def normalize(self):
        print("=====================================")
        print("=          normalization            =")
        print("=       using min max scaler        =")
        print("=====================================")
        # min max scaler for each feature
        scaler = MinMaxScaler()
        # fit the scalar with merged data
        scaler.fit(pd.concat([self.train_data, self.test_data]))
        train_norm = scaler.transform(self.train_data)
        test_norm = scaler.transform(self.test_data)

        self.train_data = pd.DataFrame(train_norm, columns=self.train_data.columns)
        self.test_data = pd.DataFrame(test_norm, columns=self.test_data.columns)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def process(self):
        self.drop_useless()
        self.categorical_processing()
        
        # impute the missing values
        my_imputer = Imputer(self.train_data, self.test_data, 1)
        my_imputer.process()
        self.train_data, self.test_data = my_imputer.get_train_data(), my_imputer.get_test_data()

        self.normalize()
        print("Preprocessing Completed")
        self.data_details()