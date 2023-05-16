#!/bin/bash

DATA_PATH=project/resources/data/glue
EXP_PATH=project/resources/output/glue
DATASETS=( "mnli" )
LM_NAMES=( "bert-base-uncased" "distilbert-base-uncased" "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" )
# LM_NAMES=( "roberta-base" )
EPOCHS=1
FEW_SHOT_STEPS=500
BATCH_SIZE=16
LR=2e-5
TRAIN_TYPE=tuned

POOLINGS=( "first" )
CLASSIFIER="linear"
SEEDS="0 42 515 1117 12345"

num_exp=0
num_err=0

# iterate over pooling strategies
for pls_idx in "${!POOLINGS[@]}"; do
  # iterate over datasets
  for dt_idx in "${!DATASETS[@]}"; do
    dataset=${TASKS[$dt_idx]}
    # iterate over encoders
    for lm_idx in "${!LM_NAMES[@]}"; do
      lm_name=${LM_NAMES[$lm_idx]}
      pooling=${POOLINGS[$pls_idx]}
      data_dir=$DATA_PATH
      echo "Few-shot Training: $lm_name ($pooling) for dataset $dataset."

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

      mkdir -p $EXP_PATH/$dataset/model_selection_results/fewshot-${FEW_SHOT_STEPS}/
      exp_dir=$EXP_PATH/$dataset/model_selection_results/fewshot-${FEW_SHOT_STEPS}/model${enc_idx}-${pooling}-${CLASSIFIER}
      # check if experiment already exists
      if [ -f "$exp_dir/classify.log" ]; then
        echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
      # if experiment is new, train classifier
      else
        # train classifier
        echo "Training ${CLASSIFIER}-classifier using '${encoder}' ($pooling) and random seed ${seed} on ${task}."
        python classify.py \
          --task "sequence_classification" \
          --train_path ${train_path} \
          --test_path ${valid_path} \
          --exp_path ${exp_dir} \
          --train_type ${TRAIN_TYPE} \
          --epochs ${EPOCHS} \
          --earlystop_ratio ${EARLYSTOP_RATIO} \
          --batch_size ${BATCH_SIZE} \
          --learning_rate ${LR} \
          --embedding_model ${encoder} \
          --pooling ${pooling} \
          --classifier ${CLASSIFIER} \
          --seed ${seed} \
          --embedding_tuning

        if [ $? -ne 0 ]; then
          echo "[Error] Training previous model. Skipping validation."
          (( num_err++ ))
        fi

        # save experiment info
        echo "${encoder} -> ${pooling} -> ${CLASSIFIER} with RS=${seed}" > $exp_dir/experiment-info.txt
      fi

      # iterate over validation paths
      pred_path=${exp_dir}/$(basename ${valid_path%.*})-pred.csv

      # run prediction
      echo "Predicting '${valid_path}' using '${exp_dir}'."
      python classify.py \
        --task "sequence_classification" \
        --train_path ${train_path} \
        --test_path ${valid_path} \
        --train_type ${TRAIN_TYPE} \
        --exp_path ${exp_dir} \
        --embedding_model ${encoder} \
        --pooling ${pooling} \
        --classifier ${CLASSIFIER} \
        --prediction_only

      # run evaluation
      echo "Evaluating '${valid_path}'."
      python evaluate.py \
        --gold_path ${valid_path} \
        --pred_path ${pred_path} \
        --out_path ${exp_dir}
      (( num_exp++ ))


      rm -rf ${exp_dir}/best.pt

      done
      echo
    done
  done
done


echo "Completed $num_exp runs with $num_err error(s)."