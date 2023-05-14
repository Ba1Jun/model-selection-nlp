#!/bin/bash

DATA_PATH=project/resources/data/glue
EXP_PATH=project/resources/output/
TASKS=( "mnli" )
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EPOCHS=1
EARLYSTOP_RATIO=1
BATCH_SIZE=16
LR=5e-6
TRAIN_TYPE=tuned

EMB_TYPES=( "transformer+cls" )
POOLINGS=( "first" )
CLASSIFIER="mlp"
SEEDS=( 42 )

num_exp=0
num_err=0
# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over pooling strategies
  for pls_idx in "${!POOLINGS[@]}"; do
    # iterate over tasks
    for tsk_idx in "${!TASKS[@]}"; do
      task=${TASKS[$tsk_idx]}
      # iterate over encoders
      for enc_idx in "${!ENCODERS[@]}"; do
        encoder="${EMB_TYPES[$pls_idx]}:${ENCODERS[$enc_idx]}"
        pooling=${POOLINGS[$pls_idx]}
        seed=${SEEDS[$rsd_idx]}
        data_dir=$DATA_PATH
        echo "Experiment: '$encoder' ($pooling) for task '$task' using seed $seed."

        # point to data dir with appropriate SEP token
        if [[ ${ENCODERS[$enc_idx]} == "roberta-base" ]] || [[ ${ENCODERS[$enc_idx]} == "cardiffnlp/twitter-roberta-base" ]]; then
          data_dir=$data_dir/roberta
        else
          data_dir=$data_dir/bert
        fi

        # set up training and validation paths
        train_path=$data_dir/$task-train.csv
        valid_paths=( $data_dir/$task-validation.csv )
        # special case: MNLI
        if [[ $task == "mnli" ]]; then
          valid_paths=( $data_dir/$task-validation_matched.csv )
        fi

        mkdir -p $EXP_PATH/$task/earlystopping-${EARLYSTOP_RATIO}/$TRAIN_TYPE
        exp_dir=$EXP_PATH/$task/earlystopping-${EARLYSTOP_RATIO}/$TRAIN_TYPE/model${enc_idx}-${pooling}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
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
            --test_path ${valid_paths[0]} \
            --exp_path ${exp_dir} \
            --train_type ${TRAIN_TYPE} \
            --epochs ${EPOCHS} \
            --earlystop_ratio ${EARLYSTOP_RATIO} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LR} \
            --embedding_model ${encoder} \
            --pooling ${pooling} \
            --classifier ${CLASSIFIER} \
            --seed ${seed}

          if [ $? -ne 0 ]; then
            echo "[Error] Training previous model. Skipping validation."
            (( num_err++ ))
          fi

          # save experiment info
          echo "${encoder} -> ${pooling} -> ${CLASSIFIER} with RS=${seed}" > $exp_dir/experiment-info.txt
        fi

        # iterate over validation paths
        for vld_idx in "${!valid_paths[@]}"; do
          valid_path=${valid_paths[$vld_idx]}
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
done

echo "Completed $num_exp runs with $num_err error(s)."