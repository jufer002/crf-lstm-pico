import jsonlines
import transform
import gluonnlp as nlp


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
            tokens = sent.split(' ')
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
    label, ind1, ind2, text_tokens = x
    data = vocab[text_tokens]   ## map tokens (strings) to unique IDs
    data = data[:max_len]       ## truncate to max_len
    return label, ind1, ind2, data


def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [ _preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


def load_dataset(train_file, test_file, max_length=250):
    # Load data from files.
    train_array = load_jsonl_to_array(train_file)
    test_array = load_jsonl_to_array(test_file)

    # Collect vocabulary and all label-types.
    vocabulary, all_labels = build_vocabulary(train_array, test_array)

    # Initialize the data transformer.
    basic_transform = transform.BasicTransform(all_labels, max_length)




if __name__ == '__main__':
    load_dataset('data/train.jsonl', 'data/test.jsonl')
