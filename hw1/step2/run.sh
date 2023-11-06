#!/bin/bash
dataset_folder="../dataset"
log_file_path="../result"
declare -a dataset_arr=("datasetA.data" "datasetB.data" "datasetC.data")
declare -a task_arr=(1 2)
# declare -a sup_arrA=(0.002 0.005 0.01)
# declare -a sup_arrB=(0.0015 0.002 0.005)
# declare -a sup_arrC=(0.01 0.02 0.03)

declare -a sup_arrA=(0.2 0.5 0.1)
declare -a sup_arrB=(0.5 0.2 0.5)
declare -a sup_arrC=(0.1 0.2 0.3)

## now loop through the above array
for task in "${task_arr[@]}"
do
    for sup_idx in 0 1 2
    do
        for dataset in "${dataset_arr[@]}"
        do
            if [ $dataset == 'datasetA.data' ]
            then
                sup_arr=("${sup_arrA[@]}")
            elif [ $dataset == 'datasetB.data' ]
            then
                sup_arr=("${sup_arrB[@]}")
            else [ $dataset == 'datasetC.data' ]
                sup_arr=("${sup_arrC[@]}")
            fi
            data_path=$dataset_folder/$dataset
            sup="${sup_arr[$sup_idx]}"
            echo "running $dataset on task $task with support: $sup"
            time python apriori.py -f $data_path -t $task -s $sup | tee -a $log_file_path/result.log
        done
    done
done