#!/bin/bash

DATA_PATH=project/resources/data/glue
# DATASETS=( "mnli" "qnli" "rte" )
DATASETS=( "qnli" )
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EMB_TYPE="transformer+cls"
POOLINGS=("first")
METHOD="NLEEP"

for pl_idx in "${!POOLINGS[@]}"; do
  POOLING=${POOLINGS[$pl_idx]}
  # iterate over datasets
  for dt_idx in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$dt_idx]}
    # create output path
    output_path=project/resources/output/$dataset
    if [ ! -d "$output_path" ]; then
      mkdir -p $output_path/encoded_dataset
      mkdir -p $output_path/results
    fi
    rm -rf $output_path/results/${METHOD}_${POOLING}.txt
    # iterate over encoders
    for enc_idx in "${!ENCODERS[@]}"; do
      encoder=${ENCODERS[$enc_idx]}
      data_dir=$DATA_PATH
      echo "Computing '$METHOD' using embeddings from '$EMB_TYPE:$encoder' for dataset '$dataset'."

      # point to data dir with appropriate SEP token
      if [[ $encoder == "roberta-base" ]] || [[ $encoder == "cardiffnlp/twitter-roberta-base" ]]; then
        data_dir=$data_dir/roberta
      else
        data_dir=$data_dir/bert
      fi

      # set up training and validation paths
      train_path=$data_dir/$dataset-train.csv
      valid_paths=( $data_dir/$dataset-validation.csv )
      # special case: MNLI
      if [[ $dataset == "mnli" ]]; then
  #      valid_paths=( $data_dir/$task-validation_matched.csv valid_path=$data_dir/$task-validation_mismatched.csv )
        valid_paths=( $data_dir/$dataset-validation_matched.csv )
      fi

      # iterate over validation paths
      for vld_idx in "${!valid_paths[@]}"; do
        valid_path=${valid_paths[$vld_idx]}
        # compute embeddings and model selection score
        python main.py \
          --method $METHOD \
          --task "sequence_classification" \
          --dataset $dataset \
          --train_path $train_path \
          --test_path $valid_path \
          --output_path $output_path \
          --text_column text --label_column label \
          --embedding_model ${EMB_TYPE}:${encoder} \
          --pooling ${POOLING} \
          --seed 42
        echo -e "-------------------------------END-------------------------------\n"
      done
    done
  done
done
