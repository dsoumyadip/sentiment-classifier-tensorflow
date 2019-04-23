__author__ = "Soumyadip"

import tensorflow as tf
import numpy as np
import os
import pickle
from typing import Dict
from tensorflow.python.keras.preprocessing import sequence

from src.model.lstm_model import lstm_with_attention
from src.utils.utils import clean_text

tf.logging.set_verbosity(tf.logging.INFO)


def train_and_evaluate(args: Dict) -> None:
    """
    Main function to run the train job
    :param args: Arguments passed in form of dictionary
    :return: None
    """

    resource_dir = os.path.join(args['root_path'], args['resources_path'])

    print("Loading cleaned, processed and vectorized data...")
    x_train = np.load(os.path.join(resource_dir, 'tokenized_train_corpus.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(resource_dir, 'tokenized_train_sentiments.npy'), allow_pickle=True)
    x_val = np.load(os.path.join(resource_dir, 'tokenized_val_corpus.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(resource_dir, 'tokenized_val_sentiments.npy'), allow_pickle=True)

    with open(os.path.join(resource_dir, 'dictionary.pickle'), 'rb') as file:
        dictionary = pickle.load(file)

    with open(os.path.join(resource_dir, 'reverse_dictionary.pickle'), 'rb') as file:
        reverse_dictionary = pickle.load(file)

    args['reverse_dict'] = reverse_dictionary

    # Padding with zeros
    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train,
                                     maxlen=args['sequence_length'],
                                     padding='post',
                                     value=0)
    x_val = sequence.pad_sequences(x_val,
                                   maxlen=args['sequence_length'],
                                   padding='post',
                                   value=0)

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_val.shape)

    x_len_train = np.array([min(len(x), args['sequence_length']) for x in x_train])
    x_len_test = np.array([min(len(x), args['sequence_length']) for x in x_val])

    def parser(x, length, y):
        features = {"x": x, "len": length}
        return features, y

    # Input functions
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
        dataset = dataset.shuffle(buffer_size=args['batch_size'] * 1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(args['batch_size'])
        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_val, x_len_test, y_val))
        dataset = dataset.batch(args['batch_size'])
        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    # 1. Function text_to_index() clean test sentences and convert it to array.
    def text_to_index(sentence):
        tokens = clean_text(sentence).split() # Remove punctuation characters
        return np.array([dictionary[t] if t in dictionary else dictionary['<UNK>'] for t in tokens])

    # 2. Function predictions_text_to_idx() returns tensors for serving input function.
    def predictions_text_to_idx(sentence):
        indexes = [text_to_index(sentence)]
        x = sequence.pad_sequences(indexes,
                                   maxlen=args['sequence_length'],
                                   padding='post',
                                   value=0)
        length = np.array([min(len(x), args['sequence_length']) for x in indexes])

        return x, length

    # 3. This function create input formats for prediction time.
    def encode_sentence(sentence):
        sentence_encoded, sentence_length = predictions_text_to_idx(str(sentence))
        features = {
            "x": tf.convert_to_tensor(sentence_encoded),
            "len": tf.convert_to_tensor(sentence_length)
        }
        return features

    # This functions serves predictions for incoming request.
    def serving_input_receiver_fn():
        sentence = tf.placeholder(shape=(None,), dtype=tf.string)

        features = encode_sentence(sentence)

        inputs = {
            "sentence": sentence
        }
        return tf.estimator.export.ServingInputReceiver(features, inputs)

    # Defining training properties
    lstm_classifier = tf.estimator.Estimator(model_fn=lstm_with_attention,
                                             params=args,
                                             model_dir=os.path.join(args['output_dir'], 'lstm'))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args['train_steps'])

    exporter = tf.estimator.LatestExporter('exporter', serving_input_receiver_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        start_delay_secs=args['eval_delay_secs'],
        throttle_secs=args['min_eval_frequency'],
        exporters=exporter)

    # Start training
    tf.estimator.train_and_evaluate(lstm_classifier, train_spec, eval_spec)
















