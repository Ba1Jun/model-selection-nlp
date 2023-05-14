#!/bin/bash

DATA_PATH=project/resources/data/ag_news
EMB_TYPE="transformer+cls"
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
POOLING="first"
DATASET="ag_news"
METHOD="LogME"

# create output path
output_path=project/resources/output/$DATASET
if [ ! -d "$output_path" ]; then
  mkdir -p $output_path/encoded_dataset
  mkdir -p $output_path/results
fi
rm -rf $output_path/results/${METHOD}_${POOLING}.txt

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  encoder=${ENCODERS[$enc_idx]}
  data_dir=$DATA_PATH
  echo "Computing '$METHOD' using embeddings from '$EMB_TYPE:$encoder' for dataset '$DATASET'."
  # compute embeddings and selection score
  python3 main.py \
    --method $METHOD \
    --task "sequence_classification" \
    --dataset $dataset \
    --train_path $DATA_PATH/train.csv \
    --test_path $DATA_PATH/test.csv \
    --output_path $output_path \
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} \
    --seed 42
  echo -e "-------------------------------END-------------------------------\n"
done
