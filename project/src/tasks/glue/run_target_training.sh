#!/bin/bash

DATA_PATH=project/resources/data/glue
OUTPUT_PATH=project/resources/output/glue

# DATASETS=( "mnli" "qnli" "rte" "sst2" "cola" "qqp" "wnli" "mrpc")
DATASETS=( "qnli" )
# LM_NAMES=( "albert-base-v2" "google/electra-base-discriminator" "microsoft/deberta-base")
LM_NAMES=( "microsoft/deberta-base" )

EPOCHS=5
BATCH_SIZE=16
LR=1e-5
MAX_LENGTH=384
POOLINGS=( "first" )
CLASSIFIER="linear"
SEEDS="42"

# iterate over pooling strategies
for pls_idx in "${!POOLINGS[@]}"; do
  # iterate over datasets
  for dt_idx in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$dt_idx]}
    # iterate over encoders
    for lm_idx in "${!LM_NAMES[@]}"; do
      lm_name=${LM_NAMES[$lm_idx]}
      pooling=${POOLINGS[$pls_idx]}
      data_dir=$DATA_PATH
      echo "Traget model training: $lm_name ($pooling) for dataset $dataset."

      # point to data dir with appropriate SEP token
      if [[ ${LM_NAMES[$lm_idx]} == "roberta-base" ]] || [[ ${LM_NAMES[$lm_idx]} == "distilroberta-base" ]]; then
        data_dir=$data_dir/roberta
      else
        data_dir=$data_dir/bert
      fi

      # set up training and validation paths
      train_path=$data_dir/$dataset-train.csv
      valid_path=$data_dir/$dataset-validation.csv
      # special case: MNLI
      if [[ $dataset == "mnli" ]]; then
        valid_path=$data_dir/$dataset-validation_matched.csv
      fi

      output_path=$OUTPUT_PATH/$dataset/encoded_dataset/target-model_${lm_name}_${pooling}/
      # mkdir -p $OUTPUT_PATH/$dataset/encoded_dataset/target-model_${lm_name}_${pooling}/

      # train classifier
      echo "Training ${CLASSIFIER}-classifier using '${encoder}' ($pooling) and random seed ${seed} on ${task}."
      python classify.py \
        --method "target-model" \
        --task "sequence_classification" \
        --dataset $dataset \
        --train_path ${train_path} \
        --test_path ${valid_path} \
        --output_path ${output_path} \
        --max_length ${MAX_LENGTH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR} \
        --lm_name ${lm_name} \
        --pooling ${pooling} \
        --classifier ${CLASSIFIER} \
        --seeds ${SEEDS} \
        --embedding_tuning
    done
  done
done