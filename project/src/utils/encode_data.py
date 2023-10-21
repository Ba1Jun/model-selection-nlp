#!/usr/bin/python3
import argparse
import logging
import torch
import sys

import numpy as np
from scipy.special import softmax
from typing import List, Tuple

# local imports
from project.src.utils.data import LabelledDataset
from project.src.utils.embeddings import load_embeddings, load_pooling_function, TransformerEmbeddings


def encode_dataset(dataset: LabelledDataset, args: argparse.Namespace, load_target_model=False) -> Tuple[np.ndarray, np.ndarray]:
    # load embedding model
    if load_target_model:
        embedding_model = torch.load(f"{args.output_path}/encoded_dataset/target-model_{args.target_model_lm_name}_{args.pooling}/best.pt")._emb
        embedding_model._lm.eval()
        embedding_model._static = True
        logging.info(f"Loaded target model {embedding_model}.")
    else:
        embedding_model = TransformerEmbeddings(args.lm_name, tokenized=(args.task == 'token_classification'), static=True)
        logging.info(f"Loaded {embedding_model}.")

    # set pooling function for sentence labeling tasks
    if args.task == 'token_classification':
        pooling_function = lambda x: x # return identity
        logging.info(f"Using all token-level embeddings (no pooling).")
    else:
        pooling_function = load_pooling_function(args.pooling)
        logging.info(f"Using pooling function '{args.pooling}' (sentence classification only).")
        

    # set up output embedding and label stores
    embeddings = np.zeros((len(dataset), embedding_model.emb_dim))
    labels = []

    # iterate over batches
    eidx = 0
    for bidx, (inputs, cur_labels, num_remaining) in enumerate(dataset.get_batches(args.batch_size)):
        # compute embeddings
        cur_embeddings = embedding_model.embed(inputs)  # list of numpy arrays with dim=(emb_dim, )

        # iterate over input sequences in current batch
        for sidx, sequence in enumerate(inputs):
            # case: labels over sequence
            # (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = ['l0', 'l1', ..., 'lN'])
            if type(cur_labels[sidx]) is list:
                # iterate over all token embeddings in the current sequence
                for tidx in range(len(sequence)):
                    tok_embedding = cur_embeddings[sidx][tidx]  # (emb_dim, )
                    embeddings[eidx, :] = tok_embedding
                    labels.append(cur_labels[sidx][tidx])
                    eidx += 1

            # case: one label for entire sequence
            # (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = 'l')
            else:
                seq_embedding = pooling_function(cur_embeddings[sidx])  # (seq_len, emb_dim) -> (emb_dim,)
                embeddings[eidx, :] = seq_embedding
                labels.append(cur_labels[sidx])
                eidx += 1

        # print progress
        sys.stdout.write(
                f"\r[{((bidx * args.batch_size) * 100) / len(dataset._inputs):.2f}%] Computing embeddings...")
        sys.stdout.flush()
    print("\r", end='')

    logging.info(f"Computed embeddings for {len(dataset)} items.")

    return np.array(embeddings), np.array(labels)
