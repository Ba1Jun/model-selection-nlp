#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import time
import os
import shutil
import jsonlines
import torch
import numpy as np

from sklearn.decomposition import PCA

from project.src.utils.data import LabelledDataset, sub_dataset_sampling
from project.src.utils.encode_data import encode_dataset
from project.src.utils.load_data import get_dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


LEARNING_METHODS = ['kNN', 'Logistic']

RANDOM_METHODS = ['EarlyStop', 'NLEEP']

TRUNCATE_METHODS = ["kNN", "NLEEP", "MSC", "LFC", "PARC", "RSA", "DDS"]

TARGET_MODEL_METHODS = ['DSE', 'PA', 'RSA', 'DDS']


def pca_reduction(args: argparse.Namespace, train_embeddings: np.ndarray, val_embeddings: np.ndarray=None, pca_model=None):
    if args.pca_components == train_embeddings.shape[1]:
        if val_embeddings is not None:
            return train_embeddings, val_embeddings, pca_model
        return train_embeddings, pca_model
    if pca_model is None:
        pca_model = PCA(n_components=args.pca_components, random_state=args.seed).fit(train_embeddings)
    transformed_train_embeddings = pca_model.transform(train_embeddings)[:, :args.pca_components]
    logging.info(f"Applying PCA to reduce train embeddings to {transformed_train_embeddings.shape[1]} components...")
    if val_embeddings is not None:
        transformed_val_embeddings = pca_model.transform(val_embeddings)[:, :args.pca_components]
        logging.info(f"Applying PCA to reduce val embeddings to {transformed_val_embeddings.shape[1]} components...")
        return transformed_train_embeddings, transformed_val_embeddings, pca_model
    return transformed_train_embeddings, pca_model


def main(args: argparse.Namespace):
    args.all_pca_components = sorted([int(c) for c in  args.all_pca_components], reverse=True)


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
        encoding_start_time = time.time()
        train_embeddings, train_labels = encode_dataset(train_dataset, args)
        encoding_end_time = time.time()
        encoding_time = encoding_end_time - encoding_start_time
        torch.save([train_embeddings, train_labels], encoded_train_dataset_file)
        logging.info(f"Encoded {args.dataset} train saved to {encoded_train_dataset_file}.")

        encoding_time_file = f"{args.output_path}/encoded_dataset/encoding_time_{args.pooling}.jsonl"
        with jsonlines.open(encoding_time_file, 'a') as f:
            f.write({
                "model": args.lm_name,
                "encoding_time": encoding_time,
            })
    
    
    
    # some training-based methods need validation dataset

    if any([m.split("-")[0] in LEARNING_METHODS for m in args.methods]):
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
    
    # target model-based methods need encoded dataset from target model
    if any([m.split("-")[0] in TARGET_MODEL_METHODS for m in args.methods]):

        target_encoded_train_dataset_file = f"{args.output_path}/encoded_dataset/target-model_{args.target_model_lm_name}_{args.pooling}/target-train-{args.target_model_lm_name}-{args.pooling}.pth"

        if os.path.exists(target_encoded_train_dataset_file):
            target_train_embeddings = torch.load(target_encoded_train_dataset_file)
            logging.info(f"Loaded target model encoded {args.dataset} val from {target_encoded_train_dataset_file}.")
        else:
            target_train_embeddings, _ = encode_dataset(train_dataset, args, load_target_model=True)
            torch.save(target_train_embeddings, target_encoded_train_dataset_file)
            logging.info(f"Target model encoded {args.dataset} train saved to {target_encoded_train_dataset_file}.")

        
    for m in args.methods:
        args.method = m
        logging.info(f"Computing {args.method} using embeddings from {args.lm_name} for dataset {args.dataset}.")

        # model selection methods
        if args.method.startswith("kNN"):
            from project.src.methods.knn import kNN as TransMetric
        elif args.method.startswith("Logistic"):
            from project.src.methods.logistic import Logistic as TransMetric
        elif args.method.startswith("NLEEP"):
            from project.src.methods.nleep import NLEEP as TransMetric
        elif args.method.startswith("HScoreR"):
            from project.src.methods.hscore_reg import HScoreR as TransMetric
        elif args.method.startswith("HScore"):
            from project.src.methods.hscore import HScore as TransMetric
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
        elif args.method.startswith("PACTran"):
            from project.src.methods.pactran import PACTran as TransMetric
        
        elif args.method.startswith("DSE"):
            from project.src.methods.dse import DSE as TransMetric
        elif args.method.startswith("PA"):
            from project.src.methods.pa import PA as TransMetric
        elif args.method.startswith("RSA"):
            from project.src.methods.rsa import RSA as TransMetric
        elif args.method.startswith("DDS"):
            from project.src.methods.dds import DDS as TransMetric

        
        for data_ratio in args.used_data_ratios:
            args.used_data_ratio = float(data_ratio)
            
            pca_models = [None] * len(args.seeds)
            target_pca_models = [None] * len(args.seeds)
            for pca_components in args.all_pca_components:
                args.pca_components = pca_components

                all_times = []
                all_scores = []
                for sd_idx, seed in enumerate(args.seeds):
                    args.seed = int(seed)
                    logging.info(f"running by seed {args.seed}...")

                    if args.used_data_ratio < 1 or (args.method.split("-")[0] in TRUNCATE_METHODS and train_embeddings.shape[0] > 10000):
                        args.max_data_size = int(args.used_data_ratio * train_embeddings.shape[0])
                        if args.max_data_size < args.pca_components and args.pca_components != 768:
                            args.max_data_size = args.pca_components
                        if args.method.split("-")[0] in TRUNCATE_METHODS and args.max_data_size > 10000:
                            args.max_data_size = 10000
                        used_train_embeddings, used_train_labels = sub_dataset_sampling(train_embeddings, train_labels, args.max_data_size, args.seed)
                        if args.method.split("-")[0] in TARGET_MODEL_METHODS:
                            used_target_train_embeddings, _ = sub_dataset_sampling(target_train_embeddings, train_labels, args.max_data_size, args.seed)
                    else:
                        used_train_embeddings, used_train_labels = train_embeddings, train_labels
                        if args.method.split("-")[0] in TARGET_MODEL_METHODS:
                            used_target_train_embeddings = target_train_embeddings
                    
                    # import pdb; pdb.set_trace()

                    # pca dimension reduction
                    if args.method.split("-")[0] in LEARNING_METHODS:
                        used_train_embeddings, used_val_embeddings, pca_models[sd_idx] = pca_reduction(args, used_train_embeddings, val_embeddings, pca_models[sd_idx])
                    else:
                        used_train_embeddings, pca_models[sd_idx] = pca_reduction(args, used_train_embeddings, None, pca_models[sd_idx])
                    if args.method.split("-")[0] in TARGET_MODEL_METHODS:
                        used_target_train_embeddings, target_pca_models[sd_idx] = pca_reduction(args, used_target_train_embeddings, None, target_pca_models[sd_idx])
                    
                    # import pdb; pdb.set_trace()
                    
                    start_time = time.time()
                    metric = TransMetric(args)
                    if args.method.split("-")[0] in LEARNING_METHODS:
                        score = metric.score(np.copy(used_train_embeddings), np.copy(used_train_labels), np.copy(used_val_embeddings), np.copy(val_labels))
                    elif args.method.split("-")[0] in TARGET_MODEL_METHODS:
                        score = metric.score(np.copy(used_train_embeddings), np.copy(used_target_train_embeddings))
                    else:
                        score = metric.score(np.copy(used_train_embeddings), np.copy(used_train_labels))
                    end_time = time.time()

                    all_times.append(end_time - start_time)
                    all_scores.append(score)

                if eval(args.save_results):
                    method_name = args.method +'-'+args.target_model_lm_name if args.method.split("-")[0] in TARGET_MODEL_METHODS else args.method
                    # data_size = min([args.max_data_size, train_embeddings.shape[0]])
                    embedding_size = used_train_embeddings.shape[1]
                    results_file = f"{args.output_path}/model_selection_results/{method_name}_{int(args.used_data_ratio * 100)}%_{embedding_size}_{args.pooling}.jsonl"
                    
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

    parser.add_argument('--methods', nargs='+', help='Lisy of Model selection method.')
    parser.add_argument('--method', type=str, nargs='?', help='Model selection method.')
    parser.add_argument('--all_methods', nargs='+', help='list of methods.')
    parser.add_argument('--dataset', type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--max_data_size', type=int, default=10000000, help='maximum number of instances for model selection.')
    parser.add_argument('--used_data_ratios', nargs='+', help='list of maximum ratio of used data for model selection.')
    parser.add_argument('--task', choices=['sequence_classification', 'token_classification'],
                        help='''Specify the type of task. Token classification requires pre-tokenized text and one 
                        label 
                        per token (both separated by space). Sequence classification requires pooling to reduce a 
                        sentence's token embeddings to one embedding per sentence.
                        ''')
    parser.add_argument('--train_path', type=str, nargs='?', help='Path to the training set.')
    parser.add_argument('--test_path', type=str, nargs='?', help='Path to the test set.')
    parser.add_argument('--output_path', type=str, nargs='?', help='Path to the output files.')

    parser.add_argument('--save_results', type=str, nargs='?', help='Whether to save results.')

    parser.add_argument('--text_column', type=str, nargs='?', help='Indicate which column to use for features.')
    parser.add_argument('--label_column', type=str, nargs='?', help='Indicate which column to use for gold labels.')

    parser.add_argument('--lm_name', type=str, nargs='?', help='pretrained language model identifier.')
    parser.add_argument('--target_model_lm_name', type=str, nargs='?', help='pretrained language model identifier of target model.')
    parser.add_argument('--pooling', choices=['mean', 'first'],
                        help='pooling strategy for sentence classification (default: None)')
    parser.add_argument('--pca_components', type=int, default=768, help='number of PCA components (default: 0, disabled)')
    parser.add_argument('--all_pca_components', nargs='+', help='list of pca components.')
    # additional settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seeds', nargs='+', help='list of random seeds')

    main(parser.parse_args())
