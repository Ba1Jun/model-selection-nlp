#!/usr/bin/python3

import argparse
import logging
import os
import sys
import time
import random
import jsonlines
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, matthews_corrcoef

from project.src.classification import load_classifier
from project.src.utils.data import LabelledDataset, sub_dataset_sampling
from project.src.utils.embeddings import load_embeddings, load_pooling_function, TransformerEmbeddings
# local imports
from project.src.utils.load_data import get_dataset


GLOBAL_STEPS = 0
FEW_SHOT_DONE = False

log_format = '%(message)s'
log_level = logging.INFO
logging.basicConfig(format=log_format, level=log_level)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Classifier Training')

    # data setup
    arg_parser.add_argument('--train_path', help='path to training data')
    arg_parser.add_argument('--test_path', help='path to validation data')
    arg_parser.add_argument('--dataset', help='name of HuggingFace dataset')
    arg_parser.add_argument('--max_length', type=int, default=512, help='maximum number of tokens in text.')
    arg_parser.add_argument('--max_data_size', type=int, default=-1, help='maximum number of instances for model selection.')
    arg_parser.add_argument('--task', choices=['sequence_classification', 'token_classification'],
                            help='''Specify the type of task. Token classification requires pre-tokenized text and 
                            one label per token (both separated by space). Sequence classification requires pooling 
                            to reduce a sentence's token embeddings to one embedding per sentence.''')
    arg_parser.add_argument('--special_tokens', nargs='*', help='special tokens list')
    arg_parser.add_argument('--text_column', default='text', help='column containing input features')
    arg_parser.add_argument('--label_column', default='label', help='column containing gold labels')

    # embedding model setup
    arg_parser.add_argument('--lm_name', type=str, nargs='?', help='pretrained language model identifier')
    # arg_parser.add_argument('--embedding_model', required=True, help='embedding model identifier')
    arg_parser.add_argument('--pooling', help='pooling strategy for sentence classification (default: None)')
    arg_parser.add_argument('--embedding_tuning', action='store_true', default=False,
                            help='set flag to tune the full model including embeddings (default: False)')
    
    # classifier setup
    arg_parser.add_argument('--classifier', required=True, help='classifier identifier')
    arg_parser.add_argument('--prediction_only', action='store_true', default=False,
                            help='set flag to run prediction on the validation data and exit (default: False)')

    # experiment setup
    arg_parser.add_argument('--method', type=str, nargs='?', help='Model selection method.')
    arg_parser.add_argument('--output_path', type=str, nargs='?', help='Path to the output files.')
    arg_parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    arg_parser.add_argument('--early_stop', type=int, default=3,
                            help='maximum number of epochs without improvement (default: 3)')
    arg_parser.add_argument('--early_stop_steps', type=int, default=0, help='maximum steps of few-shot training')
    arg_parser.add_argument('--batch_size', type=int, default=32,
                            help='maximum number of sentences per batch (default: 32)')
    arg_parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    arg_parser.add_argument('--seeds', nargs='+', help='list of random seeds')

    return arg_parser.parse_args()


def run(classifier, criterion, optimizer, dataset, batch_size, mode='train', return_predictions=False, early_stop_steps=0):
    stats = defaultdict(list)

    # set model to training mode
    if mode == 'train':
        classifier.train()
        batch_generator = dataset.get_shuffled_batches
    # set model to eval mode
    elif mode == 'eval':
        classifier.eval()
        batch_generator = dataset.get_batches

    global GLOBAL_STEPS
    global FEW_SHOT_DONE
    all_pred_labels = []
    all_true_labels = []

    # iterate over batches
    for bidx, batch_data in enumerate(batch_generator(batch_size)):
        # set up batch data
        sentences, labels, num_remaining = batch_data

        # when training, perform both forward and backward pass
        if mode == 'train':
            # zero out previous gradients
            optimizer.zero_grad()

            # forward pass
            predictions = classifier(sentences)

            # propagate loss
            loss = criterion(predictions['flat_logits'], labels)
            loss.backward()
            optimizer.step()

            # record global steps
            GLOBAL_STEPS += 1

        # when evaluating, perform forward pass without gradients
        elif mode == 'eval':
            with torch.no_grad():
                # forward pass
                predictions = classifier(sentences)
                # calculate loss
                loss = criterion(predictions['flat_logits'], labels)

        # calculate accuracy
        all_pred_labels.append(predictions['pred_labels'].detach().cpu().numpy())
        all_true_labels.append(labels)
        cur_acc = accuracy_score(np.concatenate(all_true_labels, 0), np.concatenate(all_pred_labels, 0))

        # store statistics
        stats['loss'].append(float(loss.detach().cpu().item()))

        # store predictions
        if return_predictions:
            # iterate over inputs items
            for sidx in range(predictions['labels'].shape[0]):
                # append non-padding predictions as list
                predicted_labels = predictions['labels'][sidx]
                stats['predictions'].append(predicted_labels[predicted_labels != -1].tolist())

        # print batch statistics
        pct_complete = (1 - (num_remaining / len(dataset._inputs))) * 100
        sys.stdout.write(
                f"\r[{mode.capitalize()} | Batch {bidx + 1} | {pct_complete:.2f}%] "
                f"Acc: {cur_acc:.4f}, Loss: {np.mean(stats['loss']):.4f}"
                )
        sys.stdout.flush()

        if early_stop_steps != 0 and early_stop_steps == GLOBAL_STEPS:
            logging.info("")
            logging.info(f"Few-shot training completed at {GLOBAL_STEPS} steps.")
            FEW_SHOT_DONE = True
            break
    
    all_pred_labels = np.concatenate(all_pred_labels, 0)
    all_true_labels = np.concatenate(all_true_labels, 0)
    
    stats['acc'] = accuracy_score(all_true_labels, all_pred_labels)
    stats['mcc'] = matthews_corrcoef(all_true_labels, all_pred_labels)
    stats['loss'] = np.mean(stats['loss'])

    return stats


def main():
    args = parse_arguments()

    # load pre-trained model for prediction
    if args.prediction_only:
        logging.info(f"Running in prediction mode (no training).")
        classifier_path = os.path.join(args.exp_path, 'best.pt')
        if not os.path.exists(classifier_path):
            logging.error(f"[Error] No pre-trained model available in '{classifier_path}'. Exiting.")
            exit(1)
        classifier = classifier_constructor.load(
            classifier_path, classes=label_types,
            emb_model=embedding_model, emb_pooling=pooling_function, emb_tuning=args.embedding_tuning
        )
        logging.info(f"Loaded pre-trained classifier from '{classifier_path}'.")

        stats = run(
            classifier, criterion, None, valid_data,
            args.batch_size, mode='eval', return_predictions=True
        )
        # convert label indices back to string labels
        idx_lbl_map = {idx: lbl for idx, lbl in enumerate(label_types)}
        pred_labels = [
            [idx_lbl_map[p] for p in preds]
            for preds in stats['predictions']
        ]
        pred_data = LabelledDataset(valid_data._inputs, pred_labels)
        pred_path = os.path.join(args.exp_path, f'{os.path.splitext(os.path.basename(args.test_path))[0]}-pred.csv')
        pred_data.save(pred_path)
        logging.info(f"Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f} (mean over batches).")
        logging.info(f"Saved results from {pred_data} to '{pred_path}'. Exiting.")

        exit()
    
    all_scores = []
    all_times = []
    global FEW_SHOT_DONE
    global GLOBAL_STEPS
    dataset_truncated = False
    for seed in args.seeds:
        args.seed = int(seed)
        start_time = time.time()
        # set random seeds
        if args.seed is not None:
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.random.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
        
        # TODO HuggingFace Datasets integration
        train_sentences, train_labels, valid_sentences, valid_labels = get_dataset(args)


        if args.max_data_size != -1 and len(train_sentences) > args.max_data_size:
            train_sentences, train_labels = sub_dataset_sampling(np.array(train_sentences), np.array(train_labels), args.max_data_size, args.seed)
            train_sentences, train_labels = train_sentences.tolist(), train_labels.tolist()
            dataset_truncated = True

        # setup data
        train_data = LabelledDataset(inputs=train_sentences, labels=train_labels)
        logging.info(f"Loaded {train_data} (train).")
        valid_data = LabelledDataset(inputs=valid_sentences, labels=valid_labels)
        logging.info(f"Loaded {valid_data} (dev).")
        # gather labels
        if set(train_data.get_label_types()) < set(valid_data.get_label_types()):
            logging.warning(f"[Warning] Validation data contains labels unseen in the training data.")
        label_types = sorted(set(train_data.get_label_types()) | set(valid_data.get_label_types()))

        # load embedding model
        embedding_model = TransformerEmbeddings(args.lm_name, cls=True, tokenized=(args.task == 'token_classification'), static=(not args.embedding_tuning), special_tokens=args.special_tokens, max_length=args.max_length)
        logging.info(f"Loaded {embedding_model}.")

        # load pooling function for sentence labeling tasks
        pooling_function = None
        if args.pooling is not None:
            pooling_function = load_pooling_function(args.pooling)
            logging.info(f"Applying pooling function '{args.pooling}' to token embeddings.")

        # load classifier and loss constructors based on identifier
        classifier_constructor, loss_constructor = load_classifier(args.classifier)

        # setup classifier
        classifier = classifier_constructor(
                emb_model=embedding_model, emb_pooling=pooling_function, emb_tuning=args.embedding_tuning,
                classes=label_types
                )
        logging.info(f"Using classifier:\n{classifier}")

        # setup loss
        criterion = loss_constructor(label_types)
        logging.info(f"Using criterion {criterion}.")

        # setup optimizer
        optimizer = torch.optim.AdamW(params=classifier.get_trainable_parameters(), lr=args.learning_rate)
        logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")

        # main loop
        stats = defaultdict(list)
        main_metric = 'acc'
        if args.dataset == 'cola':
            main_metric = 'mcc'
        for ep_idx in range(args.epochs):
            # iterate over training batches and update classifier weights
            ep_stats = run(
                    classifier, criterion, optimizer, train_data, args.batch_size, mode='train', early_stop_steps=args.early_stop_steps
                    )
            # print statistics
            logging.info(
                    f"[Epoch {ep_idx + 1}/{args.epochs}] Train completed with "
                    f"Acc: {ep_stats['acc']:.4f}, Mcc: {ep_stats['mcc']:.4f}, Loss: {ep_stats['loss']:.4f}"
                    )

            # iterate over batches in dev split
            ep_stats = run(
                    classifier, criterion, None, valid_data, args.batch_size, mode='eval'
                    )

            # store and print statistics
            for stat in ep_stats:
                stats[stat].append(ep_stats[stat])
            logging.info(
                    f"[Epoch {ep_idx + 1}/{args.epochs}] Validation completed with "
                    f"Acc: {ep_stats['acc']:.4f}, Mcc: {ep_stats['mcc']:.4f}, Loss: {ep_stats['loss']:.4f}"
                    )
            cur_main_metric = stats[main_metric][-1]

            # save best model
            if args.early_stop_steps==0 and cur_main_metric >= max(stats[main_metric]):
                args.output_path = f"project/resources/output/glue/{args.dataset}/encoded_dataset/target-model_{args.lm_name.split('/')[-1]}_{args.pooling}"
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)
                path = os.path.join(args.output_path, 'best.pt')
                classifier.save(path)
                logging.info(f"Saved model with best {main_metric} {cur_main_metric:.4f} to '{path}'.")

            # check for early stopping
            if (ep_idx - stats[main_metric].index(max(stats[main_metric]))) >= args.early_stop:
                logging.info(f"No improvement since {args.early_stop} epochs ({max(stats[main_metric]):.4f} loss). Early stop.")
                break
            
            # check for early stopping
            
            if FEW_SHOT_DONE:
                break

        logging.info(f"Training completed after {ep_idx + 1} epochs.")

        end_time = time.time()

        all_times.append(end_time - start_time)
        all_scores.append(max(stats[main_metric]))

        FEW_SHOT_DONE = False
        GLOBAL_STEPS = 0


        
    results_file = f"{args.output_path}/{'model_selection_results' if args.early_stop_steps!=0 else ''}/{args.method}-{args.early_stop_steps if args.early_stop_steps!=0 else args.epochs}_{args.max_data_size if dataset_truncated else len(train_sentences)}_{embedding_model.emb_dim}_{args.pooling}.jsonl"
    
    with jsonlines.open(results_file, 'a') as f:
        f.write({
            "model": args.lm_name,
            "avg_score": np.mean(all_scores),
            "avg_time": np.mean(all_times),
            "all_scores": all_scores,
            "all_times": all_times,
        })

    logging.info(f"{args.method}-{args.early_stop_steps if args.early_stop_steps!=0 else args.epochs}:")
    logging.info(f"all scores: {all_scores}")
    logging.info(f"all times: {all_times}")
    logging.info(f"avg score: {np.mean(all_scores)}, avg time: {round(np.mean(all_times), 4)}s")


if __name__ == '__main__':
    main()
