#!/bin/bash

# This script preprocesses the data for the training and testing of the model.
cd preprocessing

# CLI arguments

# common
# --dataset_root /scratch-shared/scur0402/AMOS_2022 --dataset_code amos000 --save_root /scratch-shared/scur0402/ --num_workers 4 --dataset_type AMOS
tch-shared/scur0402/ --num_workers 4 --dataset_type AMOS                                                           
CLASSES=("liver", "right kidney", "spleen", "left kidney")
DATASET_ROOTS_ROOT="/scratch-shared/scur0402" 
SAVE_ROOTS_ROOT="/scratch-shared/scur0402"
NUM_WORKERS=8
SEED=42
TEST_RATIO=0.2
VAL_RATIO=0.2
# dataset-specific
DATASET_ROOTS=("AMOS2022" "BRATS2021")
DATASET_CODES=("amos_mrct_42" "brats_mr_42")
SAVE_ROOTS=$DATASET_ROOTS
DATASET_TYPES=("AMOS" "BRATS")

# iterate 4 times and preprocess each dataset
# with the respective dataset-specific arguments
for i in {0..3}
do
    echo "python preprocess.py" \
        "--dataset_root $DATASET_ROOTS_ROOT/${DATASET_ROOTS[i]} "\
        "--save_root $SAVE_ROOTS_ROOT/${SAVE_ROOTS[i]} "\
        "--dataset_code ${DATASET_CODES[i]} "\
        "--dataset_type ${DATASET_TYPES[i]} "\
        "--classes ${CLASSES[@]} "\
        "--num_workers $NUM_WORKERS "\
        "--seed $SEED "\
        "--test_ratio $TEST_RATIO "\
        "--val_ratio $VAL_RATIO"
    python preprocess.py \
        --dataset_root $DATASET_ROOTS_ROOT/${DATASET_ROOTS[i]} \
        --save_root $SAVE_ROOTS_ROOT/${SAVE_ROOTS[i]} \
        --dataset_code ${DATASET_CODES[i]} \
        --dataset_type ${DATASET_TYPES[i]} \
        --classes ${CLASSES[@]} \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --test_ratio $TEST_RATIO \
        --val_ratio $VAL_RATIO
    if [ $? -ne 0 ]; then
        echo "Preprocessing failed for ${DATASET_ROOTS[i]}"
        exit 1
    fi
done
