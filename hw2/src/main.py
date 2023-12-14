from dataloader import DataLoader
from preprocessing import Preprocessing
from imputer import Imputer
from feature_selector import FeatureSelector
from optparse import OptionParser

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-d', '--data_path', dest='data_path', default='../dataset', help='path to data')

    (options, args) = optparser.parse_args()
    
    # load the data
    data_path = options.data_path
    my_data_loader = DataLoader(data_path + '/')
    train_data, train_label, test_data = my_data_loader.process()

    # preprocessing
    my_preprocessing = Preprocessing(train_data, train_label, test_data)
    my_preprocessing.process()
    train_data, test_data = my_preprocessing.get_train_data(), my_preprocessing.get_test_data()

    # feature selection
    my_feature_selector = FeatureSelector(train_data, train_label, test_data)
    print("Feature Selection Started")
    best_params, best_features = my_feature_selector.selector()