#!/usr/bin/python3

import argparse
import logging
import sys
import os
import torch

# from dotenv import load_dotenv
import numpy as np
# from project.src.preprocessing.tokenize import tokenize_text
from project.src.utils.data import LabelledDataset
from project.src.utils.encode_data import encode_dataset
from project.src.utils.load_data import get_dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main(args: argparse.Namespace):
    # load dataset from HF or custom
    X_train, y_train, X_val, y_val = get_dataset(args)

    # create LabelledDataset object
    train_dataset = LabelledDataset(inputs=X_train, labels=y_train)
    logging.info(f"Loaded {args.dataset} train {train_dataset}.")

    # encode dataset
    encoder = args.embedding_model.split(":")[-1].split("/")[-1]
    encoded_train_dataset_file = f"{args.output_path}/encoded_dataset/train-{encoder}-{args.pooling}.pth"
    if os.path.exists(encoded_train_dataset_file):
        train_embeddings, train_labels = torch.load(encoded_train_dataset_file)
        logging.info(f"Loaded encoded {args.dataset} train from {encoded_train_dataset_file}.")
    else:
        train_embeddings, train_labels = encode_dataset(train_dataset, args)
        torch.save([train_embeddings, train_labels], encoded_train_dataset_file)
        logging.info(f"Encoded {args.dataset} train saved to {encoded_train_dataset_file}.")
    
    if args.method in ["kNN", "Logistic"]:
        val_dataset = LabelledDataset(inputs=X_val, labels=y_val)
        logging.info(f"Loaded {args.dataset} val {val_dataset}.")

        encoded_val_dataset_file = f"{args.output_path}/encoded_dataset/val-{encoder}-{args.pooling}.pth"
        if os.path.exists(encoded_val_dataset_file):
            val_embeddings, val_labels = torch.load(encoded_val_dataset_file)
            logging.info(f"Loaded encoded {args.dataset} val from {encoded_val_dataset_file}.")
        else:
            val_embeddings, val_labels = encode_dataset(val_dataset, args)
            torch.save([val_embeddings, val_labels], encoded_val_dataset_file)
            logging.info(f"Encoded {args.dataset} val saved to {encoded_val_dataset_file}.")
    
    if args.method == "kNN":
        from project.src.methods.knn import kNN as TransMetric
    elif args.method == "Logistic":
        from project.src.methods.knn import kNN as TransMetric
    elif args.method == "NLEEP":
        from project.src.methods.nleep import NLEEP as TransMetric
    elif args.method == "HScore":
        from project.src.methods.hscore import HScore as TransMetric
    elif args.method == "HScoreR":
        from project.src.methods.hscore_reg import HScoreR as TransMetric
    elif args.method == "MSC":
        from project.src.methods.msc import MSC as TransMetric
    elif args.method == "LFC":
        from project.src.methods.lfc import LFC as TransMetric
    elif args.method == "LogME":
        from project.src.methods.logme import LogME as TransMetric
    elif args.method == "PARC":
        from project.src.methods.parc import PARC as TransMetric
    elif args.method == "GBC":
        from project.src.methods.gbc import GBC as TransMetric
    elif args.method == "TransRate":
        from project.src.methods.transrate import TransRate as TransMetric

    metric = TransMetric(regression=False)

    if args.method in ["kNN", "Logistic"]:
        score = metric.fit(train_embeddings, train_labels, val_embeddings, val_labels)
    else:
        score = metric.fit(train_embeddings, train_labels)
    
    logging.info(f"{args.method}: {score}")

    results_file = f"{args.output_path}/results/{args.method}_{args.pooling}.txt"
    with open(results_file, "a") as f:
        f.write(f"{encoder}-{args.pooling} | {args.dataset} | {args.method}: {round(score, 4)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framework for Model Selection')

    parser.add_argument('--method', type=str, nargs='?', help='Model selection method.')
    parser.add_argument('--dataset', type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--task', choices=['sequence_classification', 'token_classification'],
                        help='''Specify the type of task. Token classification requires pre-tokenized text and one 
                        label 
                        per token (both separated by space). Sequence classification requires pooling to reduce a 
                        sentence's token embeddings to one embedding per sentence.
                        ''')
    parser.add_argument('--train_path', type=str, nargs='?', help='Path to the training set.')
    parser.add_argument('--test_path', type=str, nargs='?', help='Path to the test set.')
    parser.add_argument('--output_path', type=str, nargs='?', help='Path to the output files.')

    parser.add_argument('--text_column', type=str, nargs='?', help='Indicate which column to use for features.')
    parser.add_argument('--label_column', type=str, nargs='?', help='Indicate which column to use for gold labels.')

    parser.add_argument('--embedding_model', type=str, nargs='?', help='embedding model identifier')
    parser.add_argument('--pooling', choices=['mean', 'first'],
                        help='pooling strategy for sentence classification (default: None)')
    parser.add_argument('--pca_components', type=int, default=0, help='number of PCA components (default: 0, disabled)')
    # additional settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seed', type=int, help='random seed for probabilistic components (default: None)')

    main(parser.parse_args())
