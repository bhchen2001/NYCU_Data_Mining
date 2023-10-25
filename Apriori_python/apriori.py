"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys
import time

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
            
    return itemSet, transactionList


def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    closedSet = dict()
    canNumSetBf = [len(itemSet)]
    canNumSetAf = []
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    oneCSet= returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    canNumSetAf.append(len(oneCSet))
    
    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):    
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        canNumSetBf.append(len(currentLSet))
        currentCSet= returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        # closedSet = closedSet.union(checkClosed(largeSet[k-1], currentCSet, freqSet))
        closedSet[k - 1] = checkClosed(largeSet[k-1], currentCSet, freqSet)
        canNumSetAf.append(len(currentCSet))
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    closedItems = []
    for key, value in closedSet.items():
        closedItems.extend([(tuple(item), getSupport(item)) for item in value])

    writeTask1_1(toRetItems)
    writeTask1_2(canNumSetBf, canNumSetAf)
    writeTask2(closedItems)

    return toRetItems


def printResults(items):
    """prints the generated itemsets sorted by support"""
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))

def writeTask1_1(items):
    """write the generated itemsets sorted by support to file"""
    write_line = ''
    for itemset, support in sorted(items, key=lambda x: x[1], reverse = True):
        item_str = ""
        for item in itemset:
            item_str = item_str + str(item) + ','
        item_str = item_str.strip(',')
        write_line += "{}\t{}\n".format(round(support * 100), item_str)
    with open('./result_file1.txt', mode = 'w') as write_file:
        write_file.write(write_line)

def writeTask1_2(canNumSetBf, canNumSetAf):
    write_line = str(sum(canNumSetAf)) + '\n'
    for idx in range(len(canNumSetBf)):
        write_line += "%s\t%s\t%s\n" %(str(idx + 1), str(canNumSetBf[idx]), str(canNumSetAf[idx]))
    with open('./result_file2.txt', mode = 'w') as write_file:
        write_file.write(write_line)

def checkClosed(canLevelPre, canLevelCur, freqSet):
    closedSetPre = canLevelPre
    # if the candidate in current level is empty, end the function
    # if len(canLevelCur) != 0:
    for item_pre in canLevelPre:
        # print(value)
        for item_cur in canLevelCur:
            if item_pre.issubset(item_cur) \
                and freqSet[item_pre] <= freqSet[item_cur]:
                closedSetPre.remove(item_pre)

    # print(closedSetPre)
    return closedSetPre

def writeTask2(closedItems):
    write_line = str(len(closedItems)) + '\n'
    for itemset, support in sorted(closedItems, key=lambda x: x[1], reverse = True):
        item_str = ""
        for item in itemset:
            item_str = item_str + str(item) + ','
        item_str = item_str.strip(',')
        # write_line += "%.1f\t{%s}\n" %(support, item_str)
        write_line += "{}\t{}\n".format(round(support * 100) , item_str)
    with open('./result_file3.txt', mode = 'w') as write_file:
        write_file.write(write_line)

def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record


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

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS

    start_time = time.process_time()
    items = runApriori(inFile, minSupport)
    end_time = time.process_time()

    # printResults(items)

    print("Execution Time: %f sec" %(end_time - start_time))