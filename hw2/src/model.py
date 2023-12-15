from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

import pandas as pd

class Model():
    def __init__(self, train_data, train_label, test_data, test_patient_id):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_patient_id = test_patient_id

        self.best_params = None
        self.best_features = None
        self.best_clf = None

        self.predicted_label = None

    def search_grid(self):
        # define the parameter grid of classifier and feature selector
        param_grid = {
            'sfs__k_features': [15, 20, 25],
            'clf__n_estimators': [6000, 10000],
            'clf__max_depth': [3],
            'clf__min_child_weight': [1],
            'clf__learning_rate': [0.001, 0.1],
            'clf__subsample': [0.8],
            'clf__colsample_bytree': [0.8],
            'clf__gamma': [0.1]
        }
        cv_num = 5

        # small set of parameters for testing
        if not __debug__:
            print("=====================================")
            print("=        Entering Debug Mode        =")
            print("=====================================")
            param_grid = {
                'sfs__k_features': [2, 3],
                'clf__n_estimators': [1, 2],
                'clf__max_depth': [3],
                'clf__min_child_weight': [1],
                'clf__learning_rate': [0.001],
                'clf__subsample': [0.8],
                'clf__colsample_bytree': [0.8],
                'clf__gamma': [0.1]
            }
            cv_num = 2

        print("=====================================")
        print("=          Searching Grid           =")
        print(param_grid)
        print("cv_num: ", cv_num)
        print("=====================================")

        macro_f1_scorer = make_scorer(f1_score, average='macro')

        # define the classifier and feature selector
        clf = XGBClassifier()
        sfs = SFS(clf, 
                k_features=20, 
                forward=True, 
                floating=False, 
                verbose = 3,
                scoring = macro_f1_scorer,
                n_jobs = -1,
                cv= cv_num)
        
        # create pipeline
        pipe = Pipeline([('sfs', sfs), ('clf', clf)])

        # define the grid search
        grid_search = GridSearchCV(estimator = pipe, param_grid=param_grid, cv= cv_num, scoring = macro_f1_scorer, n_jobs = -1, verbose = 3)

        # start searching
        grid_search.fit(self.train_data, self.train_label)

        # print the best parameters
        print("=====================================")
        print("=          Best Parameters          =")
        print(grid_search.best_params_)
        print("=====================================")
        
        # print the best score
        print("=====================================")
        print("=          Best Score               =")
        print(grid_search.best_score_)
        print("=====================================")

        # print the best feature
        print("=====================================")
        print("=          Best Features            =")
        print(grid_search.best_estimator_.steps[0][1].k_feature_names_)
        print("=====================================")

        # print the best classifier
        print("=====================================")
        print("=          Best Classifier          =")
        print(grid_search.best_estimator_.steps[1][1])
        print("=====================================")

        # get the best parameters
        self.best_params = grid_search.best_params_

        # get the best features from the fitted grid search
        self.best_features = []
        best_feature_idx = grid_search.best_estimator_.named_steps['sfs'].k_feature_idx_
        for feature in best_feature_idx:
            self.best_features.append(self.train_data.columns[feature])

    def cleansing(self):
        # drop the features that are not selected by best features
        self.train_data = self.train_data[self.best_features]
        self.test_data = self.test_data[self.best_features]

    def train(self):
        # fit the classifier with best parameters
        self.best_clf = XGBClassifier(n_estimators = self.best_params['clf__n_estimators'],
                                      max_depth = self.best_params['clf__max_depth'],
                                      min_child_weight = self.best_params['clf__min_child_weight'],
                                      learning_rate = self.best_params['clf__learning_rate'],
                                      subsample = self.best_params['clf__subsample'],
                                      colsample_bytree = self.best_params['clf__colsample_bytree'],
                                      gamma = self.best_params['clf__gamma'])
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