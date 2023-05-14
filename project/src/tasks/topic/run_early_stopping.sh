#!/bin/bash

DATA_PATH=project/resources/data/ag_news
EXP_PATH=project/resources/output/
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )

TASK=ag_news
EPOCHS=1
EARLYSTOP_RATIO=0.5
BATCH_SIZE=16
LR=5e-6
TRAIN_TYPE=tuned

EMB_TYPE="transformer+cls"
POOLING="first"
CLASSIFIER="mlp"
SEEDS=( 42 )

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Experiment: '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."

    mkdir -p $EXP_PATH/$TASK/earlystopping-${EARLYSTOP_RATIO}/$TRAIN_TYPE
    exp_dir=$EXP_PATH/$TASK/earlystopping-${EARLYSTOP_RATIO}/$TRAIN_TYPE/model${enc_idx}-${POOLING}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -d "$exp_dir/classify.log" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Skipped."
      continue
    fi
    echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."
    # train classifier
    python3 classify.py \
      --task "sequence_classification" \
      --train_path $DATA_PATH/train.csv \
      --test_path $DATA_PATH/dev.csv \
      --exp_path ${exp_dir} \
      --train_type ${TRAIN_TYPE} \
      --epochs ${EPOCHS} \
      --earlystop_ratio ${EARLYSTOP_RATIO} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LR} \
      --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
      --pooling ${POOLING} \
      --classifier ${CLASSIFIER} \
      --seed ${SEEDS[$rsd_idx]}

    # save experiment info
    echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${POOLING} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    # run prediction
    python classify.py \
      --task "sequence_classification" \
      --train_path $DATA_PATH/train.csv \
      --test_path $DATA_PATH/dev.csv \
      --train_type ${TRAIN_TYPE} \
      --exp_path ${exp_dir} \
      --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
      --pooling ${POOLING} \
      --classifier ${CLASSIFIER} \
      --seed ${SEEDS[$rsd_idx]} \
      --prediction_only
    
    # run evaluation
    python evaluate.py \
      --gold_path ${DATA_PATH}/dev.csv \
      --pred_path ${exp_dir}/dev-pred.csv \
      --out_path ${exp_dir}

    rm -rf ${exp_dir}/best.pt
    echo
  done
done
