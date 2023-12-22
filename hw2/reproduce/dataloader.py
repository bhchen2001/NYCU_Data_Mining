import pandas as pd
import pickle

class DataLoader():
    def __init__(self, data_path, parms_path):
        self.data_path = data_path
        self.parms_path = parms_path

    def load_data(self):
        self.train_data = pd.read_csv(self.data_path + 'train_X.csv')
        self.train_label = pd.read_csv(self.data_path + 'train_Y.csv')
        self.test_data = pd.read_csv(self.data_path + 'test_X.csv')
        self.test_patient_id = self.test_data['patient_id']
        with open(self.parms_path + 'features', 'rb') as feature_p:
            self.best_features = pickle.load(feature_p)
        with open(self.parms_path + 'parms', 'rb') as parm_p:
            self.best_parms = pickle.load(parm_p)

        self.best_features = list(self.best_features)

        # for debug mode
        if not __debug__:
            print("=====================================")
            print("=        Entering Debug Mode        =")
            print("=====================================")
            self.train_data = self.train_data.iloc[:100, :]
            self.train_label = self.train_label.iloc[:100, :]
            # self.test_data = self.test_data.iloc[:100, :]

    def get_train_data(self):
        return self.train_data
    
    def get_train_label(self):
        return self.train_label
    
    def get_test_data(self):
        return self.test_data
    
    def get_test_patient_id(self):
        return self.test_patient_id
    
    def get_best_parms(self):
        return self.best_parms
    
    def get_best_features(self):
        return self.best_features
    
    def process(self):
        self.load_data()
        print("Data Loading Completed")
        return self.get_train_data(), self.get_train_label(), self.get_test_data(), self.get_test_patient_id()