#!/bin/bash

DATA_PATH=project/resources/data/glue
# DATASETS=( "mnli" "qnli" "rte" "sst2" "cola" "qqp" "wnli" "mrpc")
DATASETS=( "mnli" )
LM_NAMES=( "bert-base-uncased" "distilbert-base-uncased" "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" )
# LM_NAMES=( "roberta-base" )
POOLING="first"
PCA_COMPONEBTS=0
# SEEDS="0 42 515 1117 12345"
SEEDS="42"

# few shot hyper-parameters
EPOCHS=100
FEW_SHOT_STEPS=200
BATCH_SIZE=16
LR=5e-6
CLASSIFIER="linear"

# METHODS=( "FewShot" "kNN" "Logistic" "NLEEP" "GBC" "HScore" "HScoreR" "LogME" "LFC" "MSC" "PARC" "SFDA" "TransRate")
METHOD="MSC"

# iterate over datasets
for dt_idx in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$dt_idx]}
  # create output path
  output_path=project/resources/output/glue/$dataset
  if [ ! -d "$output_path" ]; then
    mkdir -p $output_path/encoded_dataset
    mkdir -p $output_path/model_selection_results
  fi

  if [ $METHOD == "FewShot" ]; then
    rm -rf $output_path/model_selection_results/${METHOD}-${FEW_SHOT_STEPS}_${POOLING}.jsonl
  else
    rm -rf $output_path/model_selection_results/${METHOD}_${POOLING}.jsonl
  fi
  
  
  # iterate over plms
  for lm_idx in "${!LM_NAMES[@]}"; do
    lm_name=${LM_NAMES[$lm_idx]}
    data_dir=$DATA_PATH
    echo "Computing $METHOD using embeddings from $lm_name for dataset $dataset."

    # point to data dir with appropriate SEP token
    if [[ $lm_name == "roberta-base" ]] || [[ $lm_name == "distilroberta-base" ]]; then
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

    # compute embeddings and model selection score
    if [ $METHOD == "FewShot" ]; then
      python classify.py \
          --method $METHOD \
          --task "sequence_classification" \
          --dataset $dataset \
          --train_path ${train_path} \
          --test_path ${valid_path} \
          --output_path $output_path \
          --epochs ${EPOCHS} \
          --few_shot_steps ${FEW_SHOT_STEPS} \
          --batch_size ${BATCH_SIZE} \
          --learning_rate ${LR} \
          --lm_name ${lm_name} \
          --pooling ${POOLING} \
          --classifier ${CLASSIFIER} \
          --seeds ${SEEDS} \
          --embedding_tuning
    else
      python main.py \
        --method $METHOD \
        --task "sequence_classification" \
        --pca_components $PCA_COMPONEBTS \
        --dataset $dataset \
        --train_path $train_path \
        --test_path $valid_path \
        --output_path $output_path \
        --text_column text --label_column label \
        --lm_name ${lm_name} \
        --pooling ${POOLING} \
        --seeds ${SEEDS}
    fi
    echo -e "-------------------------------END-------------------------------\n"
  done
done

