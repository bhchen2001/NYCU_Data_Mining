#!/bin/bash
dataset_folder="./dataset"
log_file_path="./result"
declare -a dataset_arr=("datasetA.data" "datasetB.data" "datasetC.data" "datasetD.data" "datasetE.data" "datasetF.data" "datasetG.data" "datasetH.data" "datasetI.data")
declare -a sup_test=(0.05 0.1 0.2)
declare -a sup_arr=(0.001 0.0015 0.002 0.005 0.01 0.02 0.03)

## now loop through the above array
for sup_idx in 0 1 2 3 4 5 6
do
    for dataset in "${dataset_arr[@]}"
    do
        sup_arr=("${sup_arr[@]}")
        data_path=$dataset_folder/$dataset
        sup="${sup_arr[$sup_idx]}"
        echo "running $dataset with support: $sup"
        time python myEclat.py -f $data_path -s $sup | tee -a $log_file_path/result.log
    done
done