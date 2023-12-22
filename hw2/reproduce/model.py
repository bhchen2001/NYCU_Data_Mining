from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

import pandas as pd

class Model():
    def __init__(self, train_data, train_label, test_data, test_patient_id, best_parms, best_features):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_patient_id = test_patient_id
        self.best_params = best_parms
        self.best_features = best_features

        self.predicted_label = None

    def cleansing(self):
        # drop the features that are not selected by best features
        self.train_data = self.train_data[self.best_features]
        self.test_data = self.test_data[self.best_features]

    def kfold(self):
        # kfold with 5 folds, calculate marco f1 score and AUROC as the metrics
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(self.train_data):
            X_train, X_test = self.train_data.iloc[train_index], self.train_data.iloc[test_index]
            y_train, y_test = self.train_label.iloc[train_index], self.train_label.iloc[test_index]
            print("=====================================")
            print("=         KFold Details             =")
            print("=====================================")
            print("X_train: ", X_train.shape)
            print("y_train: ", y_train.shape)
            print("X_test: ", X_test.shape)
            print("y_test: ", y_test.shape)
            print("=====================================")
            print("has_died: ", y_test.sum())
            print("=====================================")
            print("Training Started")
            self.train(X_train, y_train)
            print("Training Completed")
            print("Prediction Started")
            self.predict(X_test, y_test)
            print("Prediction Completed")
            print("=====================================")
            print("=         KFold Results             =")
            print("=====================================")
            print("Macro F1 Score: ", f1_score(y_test, self.predicted_label, average='macro'))
            print("AUROC: ", roc_auc_score(y_test, self.predicted_label))
            print("=====================================")

    def train(self):
        # fit the classifier with best parameters
        print("=====================================")
        print("=         Training Details          =")
        print("selected features: ", self.best_features)
        print("selected parameters: ", self.best_params)
        print("=====================================")
        self.best_clf = XGBClassifier(n_estimators = self.best_params['clf__n_estimators'],
                                      max_depth = self.best_params['clf__max_depth'],
                                      min_child_weight = self.best_params['clf__min_child_weight'],
                                      learning_rate = self.best_params['clf__learning_rate'],
                                      subsample = self.best_params['clf__subsample'],
                                      colsample_bytree = self.best_params['clf__colsample_bytree'],
                                      gamma = self.best_params['clf__gamma'],
                                      scale_pos_weight = self.best_params['clf__scale_pos_weight'],
                                      random_state = self.best_params['clf__random_state'])
        self.best_clf.fit(self.train_data, self.train_label)
    
    def predict(self):
        # fit the classifier with train data
        self.best_clf.fit(self.train_data, self.train_label)

        # predict the test data
        self.predicted_label = self.best_clf.predict(self.test_data)
    
    def write_to_csv(self, path):
        # merge the patient id and predicted label
        df = pd.DataFrame({'patient_id': self.test_patient_id, 'pred': self.predicted_label})
        # details of the dataframe
        print("=====================================")
        print("=         Predict Details           =")
        print(df.info())
        print(df.describe())
        print(df.head())
        print("has_died: ", df['pred'].sum())
        print("=====================================")
        df.to_csv(path, index=False)