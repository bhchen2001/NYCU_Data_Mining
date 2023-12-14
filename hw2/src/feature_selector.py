import pandas as pd
from sklearn import feature_selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class FeatureSelector():
    def __init__(self, train_data, train_label, test_data):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data

    def multi_fidelity_search(self):
        # get the full train data
        train_full = self.train_data.copy()
        train_full['has_died'] = self.train_label

        # select records with has_died = 1
        train_died = train_full[train_full['has_died'] == 1]
        # select same number of records with has_died = 0
        train_alive = train_full[train_full['has_died'] == 0].sample(n=len(train_died))

        # combine the two dataframes
        train_full = pd.concat([train_died, train_alive])
        # shuffle the dataframe
        train_full = train_full.sample(frac=1)

        # get the train data and label
        partial_train_data = train_full.drop(['has_died'], axis=1)
        partial_train_label = train_full['has_died']

        return partial_train_data, partial_train_label

    def selector(self, k = 20):
        # option1: use SFS combining with XGBoost
        # define the parameter grid of classifier and feature selector
        param_grid = {
            'sfs__k_features': [30],
            'clf__n_estimators': [1000],
            'clf__max_depth': [3],
            'clf__min_child_weight': [1],
            'clf__learning_rate': [0.001],
            'clf__subsample': [0.8],
            'clf__colsample_bytree': [0.8],
            'clf__gamma': [0.1]
        }
        cv_num = 5

        # small set of parameters for testing
        if not __debug__:
            print("=====================================")
            print("=========Entering Debug Mode=========")
            print("=====================================")
            param_grid = {
                'sfs__k_features': [1, 2],
                'clf__n_estimators': [1, 2],
                'clf__max_depth': [3],
                'clf__min_child_weight': [1],
                'clf__learning_rate': [0.001],
                'clf__subsample': [0.8],
                'clf__colsample_bytree': [0.8],
                'clf__gamma': [0.1]
            }
            cv_num = 2

        macro_f1_scorer = make_scorer(f1_score, average='macro')
        
        sfs = SFS(XGBClassifier(), 
                k_features=20, 
                forward=True, 
                floating=False, 
                verbose = 3,
                scoring = macro_f1_scorer,
                cv= cv_num)
        
        # create pipeline
        pipe = Pipeline([('sfs', sfs), ('clf', XGBClassifier())])

        # define the grid search
        grid_search = GridSearchCV(estimator = pipe, param_grid=param_grid, cv= cv_num, scoring = macro_f1_scorer, n_jobs = -1, verbose = 3)

        # get partial train data and label
        partial_train_data, partial_train_label = self.multi_fidelity_search()

        # fit the grid search with data
        grid_search.fit(partial_train_data, partial_train_label)

        # get the best parameters
        best_params = grid_search.best_params_
        print("=====================================")
        print("best parameters: ", best_params)

        # get the best features from the fitted grid search
        best_features = grid_search.best_estimator_.named_steps['sfs'].k_feature_idx_
        print("best features: ", best_features)
        print("best features name: ")
        for feature in best_features:
            print(partial_train_data.columns[feature])

        return best_params, best_features