import os
import argparse
import jsonlines

from glob import iglob
from tqdm import tqdm


START_TOKEN = '<START>'
STOP_TOKEN = '<STOP>'

if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(description='Load PICO data for use in model.')

    parser.add_argument('--train_file', type=str, default='data/train.jsonl',
                        help='Output file which contains train samples from the PICO dataset.')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl',
                        help='Output file which contains test samples from the PICO dataset.')

    args = parser.parse_args()


def get_doc_id(full_path):
    filename = os.path.basename(full_path)

    # Return the id and not the extension.
    return filename.split('.')[0]


def get_doc_ids():
    filenames = iglob('data/docs/*.tokens')

    for full_path in filenames:
        yield get_doc_id(full_path)


def get_token_filepaths():
    filenames = iglob('data/docs/*.tokens')

    return filenames


def get_token_filepath(doc_id):
    return f'data/docs/{doc_id}.tokens'


def get_instance_dict(doc_id):
    annotations_path = 'data/annotations'

    # First, determine if the doc is in the train or the gold set.
    is_train = os.path.exists(f'{annotations_path}/interventions/train/{doc_id}_AGGREGATED.ann')
    if is_train:
        subdir = 'train'
    else:
        subdir = 'test/gold'

    annotation_filenames = [
        (f'{annotations_path}/{doc_type}/{subdir}/{doc_id}_AGGREGATED.ann', doc_type)
        for doc_type in ('interventions', 'outcomes', 'participants')
    ]

    all_labels = []
    for filename, doc_type in annotation_filenames:
        with open(filename) as fp:
            annotation = fp.read()

            all_labels.append([doc_type[0].upper() if ann == '1' else '0' for ann in annotation.split(',')])

    new_labels = [START_TOKEN]
    for i, o, p in zip(*all_labels):
        new_label = ''

        if i != '0':
            new_label += i
        if o != '0':
            new_label += o
        if p != '0':
            new_label += p

        elif new_label == '':
            new_label = 'X'

        new_labels.append(new_label)

    new_labels.append(STOP_TOKEN)

    # Get tokens.
    token_filepath = get_token_filepath(doc_id)
    with open(token_filepath) as fp:
        tokens = fp.read()

    instance_dict = {
        'doc_id': doc_id,
        'sent': tokens,
        'labels': ','.join(new_labels),
        'is_train': is_train,
    }

    return instance_dict


def get_instance_dicts(doc_ids):
    for doc_id in tqdm(doc_ids, desc='Reading samples from PICO set'):
        yield get_instance_dict(doc_id)


def build_gold_files():
    doc_ids = get_doc_ids()

    instance_dicts = list(get_instance_dicts(doc_ids))

    # Write to files.
    train_fp = jsonlines.open(args.train_file, 'w')
    test_fp = jsonlines.open(args.test_file, 'w')

    for instance in tqdm(instance_dicts, desc='Writing samples to gold files'):
        is_train = instance['is_train']
        if is_train:
            train_fp.write(instance)
        else:
            test_fp.write(instance)

    test_fp.close()
    train_fp.close()


# If load_data.py is run by itself, build the gold files from the PICO dataset.
if __name__ == '__main__':
    build_gold_files()
