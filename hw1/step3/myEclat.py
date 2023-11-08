import re
import time
import os
from optparse import OptionParser

class myEclat():
    def __init__(self, file_path, min_sup = 0.5):
        self.file_path = file_path
        self.result_path = './result/'
        self.min_sup = min_sup
        self.transactions = []
        self.item_dict = {}
        self.item_list = []
        self.frequent_itemset = []
        self.abs_sup = 0
        self.pruning = {1:[0, 0]}

    def loadData(self):
        with open(self.file_path) as input_file:
            for line in input_file:
                output_line = re.sub('^[0-9]* [0-9]* [0-9]* ', '', line)
                output_line = output_line.strip('\n')
                self.transactions.append([s for s in output_line.split(' ')])

    def dataPreprocessing(self):
        # scan the transaction and create dictionary for each item
        for tid in range(0, len(self.transactions)):
            for item in self.transactions[tid]:
                # print(item)
                if item in self.item_dict:
                    self.item_dict[item].append(tid)
                else:
                    self.item_dict[item] = [tid]
        # sort the dictionary by item
        self.item_dict = dict(sorted(self.item_dict.items()))
        # print(self.item_dict)
        self.abs_sup = self.min_sup * len(self.transactions)

    def frequentSingle(self):
        # get the frequent itemset with single item
        self.pruning[1][0] = len(self.item_dict.keys())
        for item in self.item_dict.keys():
            item_sup = len(self.item_dict[item])
            if item_sup >= self.abs_sup:
                self.frequent_itemset.append([item, item_sup/len(self.transactions)])
                self.item_list.append(item)
        self.pruning[1][1] = len(self.item_list)
        self.item_list = sorted(self.item_list)

    def eclat(self, pre_itemset, pre_transac, last_idx):
        if len(pre_itemset) + 1 in self.pruning:
            self.pruning[len(pre_itemset) + 1][0] += (len(self.item_list) - last_idx - 1)
        else:
            self.pruning[len(pre_itemset) + 1] = [(len(self.item_list) - last_idx - 1), 0]
        for idx in range(last_idx + 1, len(self.item_list)):
            transaction_union = pre_transac.intersection(self.item_dict[self.item_list[idx]])
            if len(transaction_union) >= self.abs_sup:
                self.pruning[len(pre_itemset) + 1][1] += 1
                new_itemset = pre_itemset + [self.item_list[idx]]
                self.frequent_itemset.append([new_itemset, len(transaction_union)/len(self.transactions)])
                self.eclat(new_itemset, transaction_union, idx)

    def writeFile(self):
        fileName1 = 'step3_task1_' + os.path.basename(self.file_path).split('.')[0] + '_' + str(self.min_sup) + '_result1.txt'
        fileName2 = 'step3_task1_' + os.path.basename(self.file_path).split('.')[0] + '_' + str(self.min_sup) + '_result2.txt'
        idx = 1
        self.frequent_itemset = sorted(self.frequent_itemset, key=lambda x: x[1], reverse=True)
        with open(fileName1, 'w') as output1:
            for item in self.frequent_itemset:
                itemset_str = str(item[0]).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                itemset_str = '{' + itemset_str + '}'
                output1.write(str(round(item[1] * 100, 1)) + '\t' + itemset_str + '\n')
        with open(fileName2, 'w') as output2:
            output2.write(str(len(self.frequent_itemset)) + '\n')
            for key, value in self.pruning.items():
                output2.write(str(idx) + '\t' + str(value[0]) + '\t' + str(value[1]) + '\n')
                idx += 1

    def process(self):
        start = time.process_time()
        self.loadData()
        self.dataPreprocessing()
        self.frequentSingle()
        for idx in range(0, len(self.item_list)):
            self.eclat([self.item_list[idx]], set(self.item_dict[self.item_list[idx]]), idx)
        self.writeFile()
        end = time.process_time()
        return end - start

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default='A.csv'
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.1,
        type="float",
    )
    (options, args) = optparser.parse_args()
    file_path = options.input
    min_sup = options.minS
    myEclat = myEclat(file_path, min_sup)
    exe_time = myEclat.process()
    print("Execution Time of Eclat (%s, sup: %s): %f sec" %(os.path.basename(file_path).split('.')[0], min_sup, exe_time))