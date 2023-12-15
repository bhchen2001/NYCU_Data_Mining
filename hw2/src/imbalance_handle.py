from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

class ImbalanceHandle():
    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def imbalance_handle(self):
        print("=====================================")
        print("=        Imbalance Handle           =")
        print("=           using SMOTE             =")
        print("=====================================")
        sm = SMOTE(random_state=42)
        rus = RandomUnderSampler(random_state=42)
        print("before SMOTE: ", self.train_data.shape, self.train_label.shape, self.train_label.sum())
        smote_train_data, smote_train_label = sm.fit_resample(self.train_data, self.train_label)
        print("after SMOTE: ", smote_train_data.shape, smote_train_label.shape, smote_train_label.sum())
        # rus_train_data, rus_train_label = rus.fit_resample(smote_train_data, smote_train_label)
        # print("after RUS: ", rus_train_data.shape, rus_train_label.shape, rus_train_label.sum())

        self.train_data = pd.DataFrame(smote_train_data, columns=self.train_data.columns)
        self.train_label = pd.DataFrame(smote_train_label, columns=self.train_label.columns)

    def get_train_data(self):
        return self.train_data
    
    def get_train_label(self):
        return self.train_label