#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import time
import os
import jsonlines
import torch
import numpy as np

from sklearn.decomposition import PCA

from project.src.utils.data import LabelledDataset
from project.src.utils.encode_data import encode_dataset
from project.src.utils.load_data import get_dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


LEARNING_METHODS = ['kNN', 'Logistic']


def pca_reduction(embeddings: np.ndarray, args: argparse.Namespace, seed: int):
    if args.pca_components == 0:
        return embeddings
    pca_model = PCA(n_components=args.pca_components, random_state=seed)
    assert len(embeddings.shape[0]) >= pca_model.n_components, \
        f"[Error] Not enough data to perform PCA ({len(embeddings.shape[0])} < {pca_model.n_components})."
    logging.info(f"Applying PCA to reduce embeddings to {pca_model.n_components} components...")
    embeddings = pca_model.fit_transform(embeddings)
    return embeddings


def main(args: argparse.Namespace):
    # load dataset from HF or custom
    X_train, y_train, X_val, y_val = get_dataset(args)

    # create LabelledDataset object
    train_dataset = LabelledDataset(inputs=X_train, labels=y_train)
    logging.info(f"Loaded {args.dataset} train {train_dataset}.")

    # encode dataset
    encoded_train_dataset_file = f"{args.output_path}/encoded_dataset/train-{args.lm_name}-{args.pooling}.pth"
    if os.path.exists(encoded_train_dataset_file):
        train_embeddings, train_labels = torch.load(encoded_train_dataset_file)
        logging.info(f"Loaded encoded {args.dataset} train from {encoded_train_dataset_file}.")
    else:
        train_embeddings, train_labels = encode_dataset(train_dataset, args)
        torch.save([train_embeddings, train_labels], encoded_train_dataset_file)
        logging.info(f"Encoded {args.dataset} train saved to {encoded_train_dataset_file}.")
    
    
    
    # some training-based methods need validation dataset
    if args.method.split("-")[0] in LEARNING_METHODS:
        val_dataset = LabelledDataset(inputs=X_val, labels=y_val)
        logging.info(f"Loaded {args.dataset} val {val_dataset}.")

        encoded_val_dataset_file = f"{args.output_path}/encoded_dataset/val-{args.lm_name}-{args.pooling}.pth"
        if os.path.exists(encoded_val_dataset_file):
            val_embeddings, val_labels = torch.load(encoded_val_dataset_file)
            logging.info(f"Loaded encoded {args.dataset} val from {encoded_val_dataset_file}.")
        else:
            val_embeddings, val_labels = encode_dataset(val_dataset, args)
            torch.save([val_embeddings, val_labels], encoded_val_dataset_file)
            logging.info(f"Encoded {args.dataset} val saved to {encoded_val_dataset_file}.")
        
        
    # model selection methods
    if args.method.startswith("kNN"):
        from project.src.methods.knn import kNN as TransMetric
    elif args.method.startswith("Logistic"):
        from project.src.methods.logistic import Logistic as TransMetric
    elif args.method.startswith("NLEEP"):
        from project.src.methods.nleep import NLEEP as TransMetric
    elif args.method.startswith("HScore"):
        from project.src.methods.hscore import HScore as TransMetric
    elif args.method.startswith("HScoreR"):
        from project.src.methods.hscore_reg import HScoreR as TransMetric
    elif args.method.startswith("MSC"):
        from project.src.methods.msc import MSC as TransMetric
    elif args.method.startswith("LFC"):
        from project.src.methods.lfc import LFC as TransMetric
    elif args.method.startswith("LogME"):
        from project.src.methods.logme import LogME as TransMetric
    elif args.method.startswith("PARC"):
        from project.src.methods.parc import PARC as TransMetric
    elif args.method.startswith("GBC"):
        from project.src.methods.gbc import GBC as TransMetric
    elif args.method.startswith("TransRate"):
        from project.src.methods.transrate import TransRate as TransMetric
    elif args.method.startswith("SFDA"):
        from project.src.methods.sfda import SFDA as TransMetric


    

    all_times = []
    all_scores = []
    for seed in args.seeds:
        args.seed = int(seed)
        logging.info(f"running by seed {args.seed}...")
        # pca dimension reduction
        train_embeddings = pca_reduction(train_embeddings, args, args.seed)
        if args.method.split("-")[0] in LEARNING_METHODS:
            val_embeddings = pca_reduction(val_embeddings, args, args.seed)
        
        start_time = time.time()
        metric = TransMetric(args)
        if args.method.split("-")[0] in LEARNING_METHODS:
            score = metric.score(np.copy(train_embeddings), np.copy(train_labels), np.copy(val_embeddings), np.copy(val_labels))
        else:
            score = metric.score(np.copy(train_embeddings), np.copy(train_labels))
        end_time = time.time()

        all_times.append(end_time - start_time)
        all_scores.append(score)

    results_file = f"{args.output_path}/model_selection_results/{args.method}_{args.pooling}.jsonl"
    with jsonlines.open(results_file, 'a') as f:
        f.write({
            "model": args.lm_name,
            "avg_score": np.mean(all_scores),
            "avg_time": np.mean(all_times),
            "all_scores": all_scores,
            "all_times": all_times,
        })

    logging.info(f"{args.method}:")
    logging.info(f"all scores: {all_scores}")
    logging.info(f"all times: {all_times}")
    logging.info(f"avg score: {np.mean(all_scores)}, avg time: {round(np.mean(all_times), 4)}s")


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

    parser.add_argument('--lm_name', type=str, nargs='?', help='pretrained language model identifier')
    parser.add_argument('--pooling', choices=['mean', 'first'],
                        help='pooling strategy for sentence classification (default: None)')
    parser.add_argument('--pca_components', type=int, default=0, help='number of PCA components (default: 0, disabled)')
    # additional settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seeds', nargs='+', help='list of random seeds')

    main(parser.parse_args())
