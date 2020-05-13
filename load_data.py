"""
Author: Julian Fernandez

This module preprocesses the gold files into dataset objects that are nearly ready to be fed into the model.
"""

import jsonlines
import transform
import gluonnlp as nlp

from build_gold_files import START_TOKEN, STOP_TOKEN


def load_jsonl_to_array(filename):
    with jsonlines.open(filename) as docs:
        array = [
            (doc['sent'], doc['labels'])
            for doc in docs
        ]

    return array


def build_vocabulary(train_array, test_array):
    """
        Inputs: arrays representing the training, validation and test data
        Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    # List of all tokens in the dataset.
    all_tokens = []
    # Keep track of all types of labels.
    all_labels = set()

    for array in (train_array, test_array):
        for i, instance in enumerate(array):
            sent, label_string = instance
            tokens = [START_TOKEN, *sent.lower().split(' '), STOP_TOKEN]
            labels = label_string.split(',')

            # In-place modification of array.
            array[i] = (tokens, labels)

            # Update running count of all tokens and all labels types.
            all_tokens.extend(tokens)
            all_labels.update(labels)

    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)

    return vocab, all_labels


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    text_tokens, labels = x

    if len(text_tokens) > max_len:
        text_tokens[max_len - 1] = STOP_TOKEN
        labels[max_len - 1] = STOP_TOKEN

        text_tokens = text_tokens[:max_len]  ## truncate to max_len
        labels = labels[:max_len]

    data = vocab[text_tokens]  ## map tokens (strings) to unique IDs

    return data, labels


def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [_preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


def load_dataset(train_file, test_file, max_length):
    # Load data from files.
    train_array = load_jsonl_to_array(train_file)
    test_array = load_jsonl_to_array(test_file)

    # Collect vocabulary and all label-types.
    vocabulary, all_labels = build_vocabulary(train_array, test_array)

    # Initialize the data transformer.
    basic_transform = transform.BasicTransform(all_labels, max_length)

    # Preprocess the data.
    train_dataset = preprocess_dataset(train_array, vocabulary, max_length)
    test_dataset = preprocess_dataset(test_array, vocabulary, max_length)

    return vocabulary, train_dataset, test_dataset, basic_transform
