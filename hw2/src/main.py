from dataloader import DataLoader
from preprocessing import Preprocessing
from imputer import Imputer
from feature_selector import FeatureSelector
from imbalance_handle import ImbalanceHandle
from model import Model
from optparse import OptionParser

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-d', '--data_path', dest='data_path', default='../dataset', help='path to data')
    optparser.add_option('-r', '--result_path', dest='result_path', default='../result/result.csv', help='path to result')

    (options, args) = optparser.parse_args()
    
    # load the data
    data_path = options.data_path
    my_data_loader = DataLoader(data_path + '/')
    train_data, train_label, test_data, test_patient_id = my_data_loader.process()

    # preprocessing & imputation
    my_preprocessing = Preprocessing(train_data, train_label, test_data)
    my_preprocessing.process()
    train_data, test_data = my_preprocessing.get_train_data(), my_preprocessing.get_test_data()

    # feature selection
    my_feature_selector = FeatureSelector(train_data, train_label, test_data)
    print("Feature Selection Started")
    fixed_features = my_feature_selector.get_fixed_features()
    print("Fixed Features Completed")
    print(fixed_features)
    best_params, best_features = my_feature_selector.selector()
    print("Feature Selection Completed")
    my_feature_selector.cleansing()
    train_data, test_data = my_feature_selector.get_train_data(), my_feature_selector.get_test_data()

    # imbalancing handle
    # my_imbalance_handle = ImbalanceHandle(train_data, train_label)
    # print("Imbalance Handle Started")
    # my_imbalance_handle.imbalance_handle()
    # print("Imbalance Handle Completed")
    # train_data, train_label = my_imbalance_handle.get_train_data(), my_imbalance_handle.get_train_label()

    # training
    my_model = Model(train_data, train_label, test_data, test_patient_id, fixed_features)
    print("Searching Started")
    my_model.search_grid()
    print("Searching Completed")
    my_model.cleansing()
    print("Training Started")
    my_model.train()
    print("Training Completed")
    print("Prediction Started")
    my_model.predict()
    print("Prediction Completed")
    my_model.write_to_csv(options.result_path)
    print("Output Completed")