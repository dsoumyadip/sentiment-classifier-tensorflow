__author__ = "Soumyadip"

import argparse
import json
import os

from . import build_model
from src.utils.utils import process_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directory paths
    parser.add_argument('--root_path',
                        help='Root directory',
                        required=True
                        )
    parser.add_argument('--train_data_path',
                        help='Train ata directory relative to root path',
                        required=True
                        )
    parser.add_argument('--val_data_path',
                        help='Validation data directory relative to root path',
                        required=True
                        )
    parser.add_argument('--resources_path',
                        help='Glove/processed data paths',
                        required=True
                        )

    # Embeddings
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        help="Word embedding size. (For glove, use 50 | 100 | 200 | 300)"
                        )

    # Model structure
    parser.add_argument("--sequence_length",
                        type=int,
                        default=200,
                        help="LSTM network length"
                        )
    parser.add_argument("--num_layers",
                        type=int,
                        default=2,
                        help="LSTM network depth"
                        )

    # Train params
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-2,
                        help="Learning rate."
                    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size."
                    )
    parser.add_argument("--keep_prob",
                        type=float,
                        default=0.8,
                        help="Dropout keep prob."
                    )
    parser.add_argument(
                        '--train_steps',
                        help='Steps to run the training job for',
                        type=int,
                        default=10000
                    )
    parser.add_argument(
                        '--eval_steps',
                        help='Number of steps to run evaluation for at each checkpoint',
                        default=10,
                        type=int
                    )
    parser.add_argument(
                        '--output_dir',
                        help='Directory to write checkpoints and export models',
                        required=True
                    )
    parser.add_argument(
                        '--job-dir',
                        help='this model ignores this field, but it is required by gcloud',
                        default='junk'
                    )

    parser.add_argument(
                        '--eval_delay_secs',
                        help='How long to wait before running first evaluation',
                        default=10,
                        type=int
                    )
    parser.add_argument(
                        '--min_eval_frequency',
                        help='Seconds between evaluations',
                        default=300,
                        type=int
                    )

    # Others
    parser.add_argument(
                        '--min_word_frequency',
                        help='Min word occurrence to keep in dict',
                        default=20,
                        type=int
                    )

    args = parser.parse_args()
    arguments = args.__dict__

    print(arguments)

    process_data(arguments['root_path'], arguments['train_data_path'], arguments['val_data_path'], arguments['resources_path'])
    build_model.train_and_evaluate(arguments)
