from dataloader import DataLoader
from preprocessing import Preprocessing
from model import Model
from optparse import OptionParser

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-d', '--data_path', dest='data_path', default='../dataset', help='path to data')
    optparser.add_option('-r', '--result_path', dest='result_path', default='../result/result.csv', help='path to result')
    optparser.add_option('-p', '--parms_path', dest='parms_path', default='./parm/1220', help='path to best parms')

    (options, args) = optparser.parse_args()
    
    # load the data
    data_path = options.data_path
    parms_path = options.parms_path
    my_data_loader = DataLoader(data_path + '/', parms_path + '/')
    train_data, train_label, test_data, test_patient_id = my_data_loader.process()
    best_parms, best_features = my_data_loader.get_best_parms(), my_data_loader.get_best_features()

    print(best_parms, best_features)

    # preprocessing & imputation
    my_preprocessing = Preprocessing(train_data, train_label, test_data)
    my_preprocessing.process()
    train_data, test_data = my_preprocessing.get_train_data(), my_preprocessing.get_test_data()

    # training
    my_model = Model(train_data, train_label, test_data, test_patient_id, best_parms, best_features)
    my_model.cleansing()
    print("Training Started")
    my_model.train()
    print("Training Completed")
    print("Prediction Started")
    my_model.predict()
    print("Prediction Completed")
    my_model.write_to_csv(options.result_path)
    print("Output Completed")