#!/bin/bash

CUDA_VISIBLE_DEVICES=0

DATA_PATH=project/resources/data/glue
DATASETS=( "cola" "sst2" "mrpc" "qqp" "mnli" "qnli" "rte"  "wnli" )
# DATASETS=( "wnli" )
LM_NAMES=( "bert-base-uncased" "distilbert-base-uncased" "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" )
# LM_NAMES=( "distilroberta-base" )
POOLING="first"
# SEEDS="0 42 515 1117 12345"
SEEDS="42"

# few shot hyper-parameters
EPOCHS=100
EARLY_STOP_STEPS=50
BATCH_SIZE=16
LR=1e-5
CLASSIFIER="linear"

# target model-based method hyper-parameters
# TARGET_LM="albert-base-v2"
TARGET_LM="deberta-base"
# TARGET_LM="electra-base-discriminator"

# ALL_PCA_COMPONEBTS="16 32 64 128 256 384 512 768"
ALL_PCA_COMPONEBTS="512"
# DATA_RATIOS="0.1 0.25 0.5 0.75"
DATA_RATIOS="1"
# METHODS=( "kNN-1-l2" "kNN-3-l2" "kNN-5-l2" "kNN-1-cos" "kNN-3-cos" "kNN-5-cos" "Logistic" "NLEEP" "GBC" "HScore" "HScoreR" "LogME" "LFC-dot" "MSC-cos" "PARC-corr" "SFDA" "TransRate" "DSE-dot" "DSE-cos" "PA" "RSA-corr" "DDS-cos" "PACTran-1-100")

# METHODS="kNN-5-corr GBC LFC-cos TransRate NLEEP Logistic PACTran-1-10"
METHODS="RSA-l2"

SAVE_RESULTS="True"



# iterate over datasets
for dt_idx in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$dt_idx]}
  # create output path
  output_path=project/resources/output/glue/$dataset
  mkdir -p $output_path/encoded_dataset
  mkdir -p $output_path/model_selection_results
  
  # iterate over plms
  for lm_idx in "${!LM_NAMES[@]}"; do
    lm_name=${LM_NAMES[$lm_idx]}
    data_dir=$DATA_PATH

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

    # # compute embeddings and model selection score
    # if [ $METHOD == "EarlyStop" ]; then
    #   python classify.py \
    #       --method $METHOD \
    #       --task "sequence_classification" \
    #       --dataset $dataset \
    #       --max_data_size $MAX_DATA_SICE \
    #       --train_path ${train_path} \
    #       --test_path ${valid_path} \
    #       --output_path $output_path \
    #       --epochs ${EPOCHS} \
    #       --early_stop_steps ${EARLY_STOP_STEPS} \
    #       --batch_size ${BATCH_SIZE} \
    #       --learning_rate ${LR} \
    #       --lm_name ${lm_name} \
    #       --pooling ${POOLING} \
    #       --classifier ${CLASSIFIER} \
    #       --seeds ${SEEDS} \
    #       --embedding_tuning
    # else
    python main.py \
      --methods $METHODS \
      --task "sequence_classification" \
      --all_pca_components $ALL_PCA_COMPONEBTS \
      --dataset $dataset \
      --used_data_ratios ${DATA_RATIOS} \
      --train_path $train_path \
      --test_path $valid_path \
      --output_path $output_path \
      --text_column text --label_column label \
      --lm_name ${lm_name} \
      --target_model_lm_name ${TARGET_LM} \
      --pooling ${POOLING} \
      --seeds ${SEEDS} \
      --save_results ${SAVE_RESULTS}
    # fi
    echo -e "-------------------------------END-------------------------------\n"
  done
done

