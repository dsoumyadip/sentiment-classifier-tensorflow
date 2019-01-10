# Importing dependencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tempfile
from utils import *
from lstm_model_fn import *
import os
import string
import numpy as np

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

# Parsing application arguments

flags = tf.app.flags

flags.DEFINE_integer("vocab_size", 5000,"Number of words in vocabulary")

flags.DEFINE_integer("sentence_size", 200,"Maximum sentence size after padding.")

flags.DEFINE_integer("embedding_size", 50, "Dimension of word embedding vector.")

flags.DEFINE_integer("training_steps", 100000 , "Number of steps to train")

flags.DEFINE_float("learning_rate", 0.1,"Learning rate")

flags.DEFINE_string("batch_size", 64, "Batch size for training")

flags.DEFINE_string("no_of_units", 200, "Number of LSTM blocks.")

FLAGS = flags.FLAGS

# Directory to save trained models and weights
model_dir = '../models'
word_embedding_path = '../resources/glove.twitter.27B.50d.txt'

# For training we are using IMDB movie review dataset.
# We are using 50 dimensional Glove word vector as embedding matrix.

print("Loading data...")
(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(num_words=FLAGS.vocab_size)
print(len(y_train), "train sequences")
print(len(y_test), "test sequences")

word_index = imdb.get_word_index()
word_inverted_index = {v: k for k, v in word_index.items()}

# The first indexes in the map are reserved to represent things other than tokens
index_offset = 3
word_inverted_index[-1 - index_offset] = '_'  # Padding at the end
word_inverted_index[1 - index_offset] = '>'  # Start of the sentence
word_inverted_index[2 - index_offset] = '?'  # OOV
word_inverted_index[3 - index_offset] = ''  # Un-used


def index_to_text(indexes):
    return ' '.join([word_inverted_index[i - index_offset] for i in indexes])


print(index_to_text(x_train_variable[15000]))

if not os.path.exists(word_embedding_path):
    raise Exception('Please download glove')


# Loading embedding matrix
embedding_matrix = load_glove_embeddings(word_embedding_path, word_index,
                                         FLAGS.vocab_size, FLAGS.embedding_size)

# Padding with zeros
print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train_variable,
                                 maxlen=FLAGS.sentence_size,
                                 padding='post',
                                 value=0)
x_test = sequence.pad_sequences(x_test_variable,
                                maxlen=FLAGS.sentence_size,
                                padding='post',
                                value=0)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

x_len_train = np.array([min(len(x), FLAGS.sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), FLAGS.sentence_size) for x in x_test_variable])


def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y


# Input functions
def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_size*1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return tf.cast(embedding_matrix, tf.float32)


params = {'embedding_initializer': my_initializer,
          'vocab_size': FLAGS.vocab_size,
          'embedding_size': FLAGS.embedding_size,
          'learning_rate': FLAGS.learning_rate,
          'batch_size': FLAGS.batch_size
}


all_classifiers = {}


def train_and_evaluate(classifier):
    all_classifiers[classifier.model_dir] = classifier
    classifier.train(input_fn=train_input_fn, steps=FLAGS.training_steps)


lstm_classifier = tf.estimator.Estimator(model_fn=FLAGS.lstm_model_fn,
                                         params=params,
                                         model_dir=os.path.join(model_dir, 'lstm'))


# Start training
train_and_evaluate(lstm_classifier)


# These functions are being used during prediction.

# 1. Function text_to_index() clean test sentences and convert it to array.
def text_to_index(sentence):
    # Remove punctuation characters except for the apostrophe
    translator = str.maketrans('', '', string.punctuation.replace("'", ''))
    tokens = sentence.translate(translator).lower().split()
    return np.array([1] + [word_index[t] + index_offset if t in word_index else 2 for t in tokens])


# 2. Function predictions_text_to_idx() returns tensors for serving input function.
def predictions_text_to_idx(sentence):
    indexes = [text_to_index(sentence)]
    x = sequence.pad_sequences(indexes,
                               maxlen=FLAGS.sentence_size,
                               padding='post',
                               value=0)
    length = np.array([min(len(x), FLAGS.sentence_size) for x in indexes])

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


# Save the model
lstm_classifier.export_savedmodel(
    os.path.join(model_dir, 'lstm'),
    serving_input_receiver_fn=serving_input_receiver_fn)



