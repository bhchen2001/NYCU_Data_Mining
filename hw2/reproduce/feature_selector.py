import pandas as pd
from sklearn import feature_selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.decomposition import PCA

class FeatureSelector():
    def __init__(self, train_data, train_label, test_data):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.best_params = None
        self.best_features = None
        self.fixed_features = None

    def pca_pair(self):
        # find the correlation between features
        feature_corr = self.train_data.corr().abs()
        # for each feature, find other features that have correlation > 0.8 with it
        # and compose a set for each correlation pair
        corr_pair = []
        for feature in feature_corr.columns:
            corr = feature_corr[feature].drop(feature)
            corr = corr[corr > 0.8]
            for f in corr.index:
                corr_pair.add(tuple(sorted([feature, f])))

    def get_fixed_features(self, f = 16):
        full_train = self.train_data.copy()
        full_train['has_died'] = self.train_label

        # calculate the correlation between each feature and the label
        # label_corr = full_train.corr()['has_died'].abs().sort_values(ascending=False)

        # calculate the point biserial correlation between each feature and the label
        label_corr = pd.Series(index = full_train.columns)
        for feature in full_train.columns:
            label_corr[feature] = stats.pointbiserialr(full_train[feature], full_train['has_died'])[0]
        label_corr = label_corr.abs().sort_values(ascending=False)

        print("=====================================")
        print("=          Label Correlation        =")
        print(label_corr)
        print("=====================================")

        # get the top f features
        top_f_features = label_corr[:f].index.tolist()
        top_f_features.remove('has_died')

        # correlation between features
        feature_corr = full_train[top_f_features].corr().abs()

        # for each feature, find the feature that has the highest correlation with it
        feature_to_remove = set()
        feature_to_keep = []
        for feature in top_f_features:
            if feature in feature_to_remove or feature in feature_to_keep:
                continue
            # correlation b/w feature and other features
            corr = feature_corr[feature].drop(feature)
            # find the feature that has the highest correlation with label
            corr = corr[corr > 0.8]
            max_feature, max_corr = feature, label_corr[feature]
            for f in corr.index:
                if label_corr[f] > max_corr:
                    max_feature = f
                    max_corr = label_corr[f]
            for f in corr.index:
                if f != max_feature:
                    feature_to_remove.add(f)
            if feature != max_feature:
                feature_to_remove.add(feature)
            if max_feature not in feature_to_keep:
                feature_to_keep.append(max_feature)

        print("=====================================")
        print("=          Fixed Features          =")
        print("length: ", len(feature_to_keep))
        print(feature_to_keep)
        print("=====================================")

        self.fixed_features = feature_to_keep
        return feature_to_keep

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

    def selector(self, k = 25):
        # option1: use SFS combining with XGBoost
        # define the parameter grid of classifier and feature selector
        param_grid = {
            'sfs__k_features': [k],
            'clf__n_estimators': [12000],
            'clf__max_depth': [3],
            'clf__min_child_weight': [1],
            'clf__learning_rate': [0.001],
            'clf__subsample': [0.8],
            'clf__colsample_bytree': [0.8],
            'clf__gamma': [0.1]
        }
        cv_num = 5

        print("=====================================")
        print("=          Searching Grid           =")
        print(param_grid)
        print("cv_num: ", cv_num)
        print("=====================================")

        # small set of parameters for testing
        if not __debug__:
            print("=====================================")
            print("=        Entering Debug Mode        =")
            print("=====================================")
            param_grid = {
                'sfs__k_features': [len(self.fixed_features) + 1, len(self.fixed_features) + 2],
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

        # find the corresponding index of fixed features
        fixed_features_index = tuple()
        for feature in self.fixed_features:
            fixed_features_index += (self.train_data.columns.get_loc(feature), )

        self.best_features = set()

        for round in range(3):
            print("=====================================")
            print("round: ", round)

            sfs = SFS(XGBClassifier(), 
                    k_features=20, 
                    forward=True, 
                    floating=False, 
                    verbose = 3,
                    scoring = macro_f1_scorer,
                    n_jobs = -1,
                    cv= cv_num,
                    fixed_features = fixed_features_index)
            
            # create pipeline
            pipe = Pipeline([('sfs', sfs), ('clf', XGBClassifier())])

            # define the grid search
            grid_search = GridSearchCV(estimator = pipe, param_grid=param_grid, cv= cv_num, scoring = macro_f1_scorer, n_jobs = -1, verbose = 3)

            # run several times to get the best features
            # get partial train data and label
            partial_train_data, partial_train_label = self.multi_fidelity_search()

            
            print("=    Size of Partial Train Data     =")
            print("data shape: ", partial_train_data.shape)

            # fit the grid search with data
            grid_search.fit(partial_train_data, partial_train_label)

            # get the best parameters
            self.best_params = grid_search.best_params_
            print("best parameters: ", self.best_params)

            # get the best features from the fitted grid search
            best_feature_idx = grid_search.best_estimator_.named_steps['sfs'].k_feature_idx_
            print("best features: ", best_feature_idx)
            print("best features name: ")
            for feature in best_feature_idx:
                self.best_features.add(partial_train_data.columns[feature])
                print(partial_train_data.columns[feature])
            print("=====================================")

        self.best_features = list(self.best_features)
        print("final best features: ", self.best_features)
        print("final best features size: ", len(self.best_features))

        return self.best_params, self.best_features
    
    def cleansing(self):
        # drop the features that are not selected
        self.train_data = self.train_data[self.best_features]
        self.test_data = self.test_data[self.best_features]

    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data