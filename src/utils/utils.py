__author__ = "Soumyadip"

import re
import os
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Tuple, Dict

SENTIMENT_TYPES = ['pos', 'neg']  # Types of outcome
MIN_WORD_OCCURRENCE = 20  # Minimum number of occurrence of a word to keep in dictionary
EMBEDDING_SIZE = 50  # Dimension of Glove embedding


def clean_text(text: str) -> str:
    """
    This functions cleans text
    :param text: String containing text
    :return: Cleaned text
    """
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


def load_data(root_dir: str, train_data_path: str, val_data_path: str) -> Tuple:
    """
    This function load raw text data from file
    :param root_dir: Root directory of project
    :param train_data_path: Training data path
    :param val_data_path: Validation data path
    :return: Tuple of loaded dataset list
    """
    train_corpus = []
    train_sentiments = []
    val_corpus = []
    val_sentiments = []

    for i in SENTIMENT_TYPES:
        files_to_read = os.listdir(os.path.join(root_dir, train_data_path, i))
        for file in files_to_read:
            text = open(os.path.join(root_dir, train_data_path, i, file), encoding='utf-8', errors='ignore').read()
            train_corpus.append(clean_text(text))
            train_sentiments.append(i)

    for i in SENTIMENT_TYPES:
        files_to_read = os.listdir(os.path.join(root_dir, val_data_path, i))
        for file in files_to_read:
            text = open(os.path.join(root_dir, val_data_path, i, file), encoding='utf-8', errors='ignore').read()
            val_corpus.append(clean_text(text))
            val_sentiments.append(i)

    return train_corpus, train_sentiments, val_corpus, val_sentiments


def process_data(root_dir: str, train_data_path: str, val_data_path: str, resources_data_path: str) -> None:
    """
    This function load data, clean data and save to disk for future use
    :param root_dir: Root directory of project
    :param train_data_path: Training data path
    :param val_data_path: Validation data path
    :param resources_data_path: Path to save processed data
    :return: None
    """

    print("Loading data from files...")
    train_corpus, train_sentiments, val_corpus, val_sentiments = load_data(root_dir, train_data_path, val_data_path)

    # Creating vocab dictionary to count occurrences of words
    word2count = {}
    for corpus in train_corpus:
        for word in corpus.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    # Creating words in vocab to integer mapping
    vocabs2int = {}
    index = 0
    tokens = ['<PAD>', '<UNK>'] # Adding tokens for padding and unknown words

    for token in tokens:
        vocabs2int[token] = len(vocabs2int) + 1

    # Filtering low occurrence words from dictionary
    for word, count in word2count.items():
        if count >= MIN_WORD_OCCURRENCE:
            vocabs2int[word] = index
            index += 1

    # Reverse dictionary
    int2vocabs = {index: word for word, index in vocabs2int.items()}

    tokenized_train_corpus = []
    tokenized_val_corpus = []

    # Vectorizing text(list of integer)
    for corpus in train_corpus:
        text2int = []
        for word in corpus.split():
            if word in vocabs2int:
                text2int.append(vocabs2int[word])
            else:
                text2int.append(vocabs2int['<UNK>'])
        tokenized_train_corpus.append(text2int)

    for corpus in val_corpus:
        text2int = []
        for word in corpus.split():
            if word in vocabs2int:
                text2int.append(vocabs2int[word])
            else:
                text2int.append(vocabs2int['<UNK>'])
        tokenized_val_corpus.append(text2int)

    tokenized_train_sentiments = []
    tokenized_val_sentiments = []

    for i in train_sentiments:
        if i == 'pos':
            tokenized_train_sentiments.append(1)
        else:
            tokenized_train_sentiments.append(0)

    for i in val_sentiments:
        if i == 'pos':
            tokenized_val_sentiments.append(1)
        else:
            tokenized_val_sentiments.append(0)

    print("Saving data to disk...")

    with open(os.path.join(resources_data_path, 'dictionary.pickle'), 'wb') as handle:
        pickle.dump(vocabs2int, handle)

    with open(os.path.join(resources_data_path, 'reverse_dictionary.pickle'), 'wb') as handle:
        pickle.dump(int2vocabs, handle)

    np.save(os.path.join(resources_data_path, 'tokenized_train_corpus'),tokenized_train_corpus)
    np.save(os.path.join(resources_data_path, 'tokenized_val_corpus'), tokenized_val_corpus)
    np.save(os.path.join(resources_data_path, 'tokenized_train_sentiments'), tokenized_train_sentiments)
    np.save(os.path.join(resources_data_path, 'tokenized_val_sentiments'), tokenized_val_sentiments)


def embedding_initializer(reverse_dict: Dict, root_dir: str, resources_dir: str, embedding_size=EMBEDDING_SIZE):
    """
    This function return glove embedding as input layer initializer
    :param reverse_dict: Reverse dictionary
    :param root_dir: Root directory path
    :param resources_dir: Directory where Glove file is present
    :param embedding_size: Dimension of Glove vector
    :return: Numpy array of Glove vector
    """

    print("Loading glove vector")
    glove_file = os.path.join(root_dir, resources_dir) + "/glove.twitter.27B.%dd.txt" % embedding_size
    print(glove_file)
    if not os.path.exists(glove_file):
        raise Exception('Please download glove.')

    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word2vec_list = list()

    for index, word in reverse_dict.items():
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word2vec_list.append(word_vec)

    return np.array(word2vec_list)








