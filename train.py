import argparse
import gluonnlp as nlp
import mxnet as mx
import copy
import json
import os

from load_data import load_dataset
from model import PicoExtractor
from mxnet import autograd
from mxnet import gluon
from tqdm import tqdm


ENT_TYPES = [
    # Participants
    'P',
    # Interventions
    'I',
    # Outcomes
    'O',
]


FULL_ENT_LABELS = {
    'P': 'PARTIC.',
    'I': 'INTERV.',
    'O': 'OUTCOME',
    'ALL': 'ALL'
}


def get_data_loader(dataset, transformer, batch_size, shuffle):
    transformed_dataset = gluon.data.ArrayDataset(dataset).transform(transformer)

    return gluon.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(label_map, ctx, vocab_size, embedding_dim, lstm_hidden_dim, vocabulary, dropout):
    pico_model = PicoExtractor(
        tag2Idx=label_map,
        ctx=ctx,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        dropout=dropout
    )
    # Initialize params
    pico_model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True, vocabulary=vocabulary)

    return pico_model


def safe_divide(a, b):
    return a / b if b != 0 else 0


def compute_metrics(pred_label_strings, true_label_strings):
    # Create a dictionary that holds metrics for each entity type.
    metric_dict = {
        ent_type: {
            'f1': 0,
            'prec': 0,
            'recall': 0
        }
        for ent_type in ENT_TYPES
    }

    # Keep a running count of the total to compute metrics over all entity types.
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Compute the metrics for each entity type.
    for ent_type in metric_dict:
        # Compute true-positives, false-positives, and false-negatives for entity types.
        tp = 0
        fp = 0
        fn = 0

        for pred, gold in zip(pred_label_strings, true_label_strings):
            if ent_type in pred and ent_type in gold:
                tp += 1
            elif ent_type in pred and ent_type not in gold:
                fp += 1
            elif ent_type not in pred and ent_type in gold:
                fn += 1

        # Update totals.
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = metric_dict[ent_type]['prec'] = safe_divide(tp, tp+fp)
        recall = metric_dict[ent_type]['recall'] = safe_divide(tp, tp+fn)
        metric_dict[ent_type]['f1'] = 2 * safe_divide(prec*recall, prec+recall)

    # Compute metrics across all entity types.
    metric_dict['ALL'] = {}
    total_prec = metric_dict['ALL']['prec'] = safe_divide(total_tp, total_tp+total_fp)
    total_recall = metric_dict['ALL']['recall'] = safe_divide(total_tp, total_tp+total_fn)
    metric_dict['ALL']['f1'] = 2 * safe_divide(total_prec*total_recall, total_prec+total_recall)

    return metric_dict


def predict_on_test(model, test_dataset, label_map):
    inv_label_map = {
        label_id: label_str
        for label_str, label_id in label_map.items()
    }

    total_metrics = {
        ent_type: {
            'f1': 0,
            'prec': 0,
            'recall': 0
        }
        for ent_type in ENT_TYPES + ['ALL']
    }

    for i, (data_out_of_context, labels_out_of_context) in tqdm(enumerate(test_dataset),
                                                                desc=f'Predicting on test',
                                                                total=len(test_dataset)):
        data = data_out_of_context.as_in_context(ctx)
        labels = labels_out_of_context.as_in_context(ctx)

        for x, y in zip(data, labels):
            score, tag_seq = model(x)

            pred_label_strings = [inv_label_map[tag] for tag in tag_seq]
            true_label_strings = [inv_label_map[label.asscalar()] for label in y]

            metric_dict = compute_metrics(pred_label_strings, true_label_strings)

            # Update total metric dict.
            for ent_type in metric_dict:
                for key in metric_dict[ent_type]:
                    total_metrics[ent_type][key] += metric_dict[ent_type][key]

    avg_dict = copy.deepcopy(total_metrics)
    for ent_type in total_metrics:
        avg_dict[ent_type]['f1'] /= len(test_dataset) * args.batch_size
        avg_dict[ent_type]['prec'] /= len(test_dataset) * args.batch_size
        avg_dict[ent_type]['recall'] /= len(test_dataset) * args.batch_size

    print(json.dumps(avg_dict, indent=2))

    return avg_dict


def evaluate(model, test_dataset):
    for i, (data_out_of_context, labels_out_of_context) in tqdm(enumerate(test_dataset),
                                                                desc=f'Evaluating on test',
                                                                total=len(test_dataset)):
        data = data_out_of_context.as_in_context(ctx)
        label = labels_out_of_context.as_in_context(ctx)


def train_model(model, train_data_loader, test_dataset, num_epochs):
    differentiable_params = []

    # Do not apply weight decay on LayerNorm and bias terms.
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

    # Train model.
    for epoch in range(num_epochs):
        epoch_loss = 0

        for i, (data_out_of_context, labels_out_of_context) in tqdm(enumerate(train_data_loader),
                                                                    desc=f'Training on epoch[{epoch}]',
                                                                    total=len(train_data_loader)):
            # Contextualize data and the label.
            data = data_out_of_context.as_in_context(ctx)
            labels = labels_out_of_context.as_in_context(ctx)

            with autograd.record():
                neg_log_likelihood = model.neg_log_likelihood(data, labels, ag=autograd)
                neg_log_likelihood.backward()

            trainer.step(1)
            epoch_loss += neg_log_likelihood.mean().asscalar()

        # epoch_results = evaluate(model, test_dataset)
        print(f'Epoch: {epoch}')
        print(f'  Loss: {epoch_loss}')

        metric_dict = predict_on_test(model, test_dataset, model.label_map)

        with open('output/metrics.txt', 'a') as fp:
            json.dump(metric_dict, fp, indent=4)
            fp.write('\n')

        if args.save_params:
            model.save_parameters('model/model.params')


if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
    parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data',
                        default='data/train.jsonl')
    parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data',
                        default='data/test.jsonl')
    parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
    parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=32)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.3)
    parser.add_argument('--embedding_source', type=str, default='glove.840B.300d',
                        help='Pre-trained embedding source name')
    parser.add_argument('--load_params', action='store_true', default=True,
                        help='If argument is present, load params from model/model.params')
    parser.add_argument('--save_params', action='store_true', default=True,
                        help='If argument is present, save params after each epoch to model/model.params')
    parser.add_argument('--no_train', action='store_true', default=False,
                        help='If argument is present, skip training and just predict on the test data.')
    parser.add_argument('--output_file', type=str, help='File in which to write test predictions.',
                        default='output/preds_test.jsonl')

    args = parser.parse_args()

    # Load the vocabulary, dataset, and data transformer.
    vocabulary, train_dataset, test_dataset, basic_transform = load_dataset(
        train_file=args.train_file, test_file=args.test_file, max_length=128
    )

    # Set the word embeddings.
    if args.embedding_source:
        glove_embeddings = nlp.embedding.create('glove', source=args.embedding_source)

        vocabulary.set_embedding(glove_embeddings)

    # Create context object.
    ctx = mx.cpu()

    # Build a model.
    lstm_hidden_dim = 16
    model = build_model(
        label_map=basic_transform.get_label_map(), ctx=ctx,
        vocab_size=len(vocabulary),
        embedding_dim=len(vocabulary.embedding.idx_to_vec[0]),
        lstm_hidden_dim=lstm_hidden_dim,
        vocabulary=vocabulary,
        dropout=args.dropout
    )

    if args.load_params:
        if os.path.exists('model/model.params'):
            print('Loading model parameters from model/model.params...')
            model.load_parameters('model/model.params', ctx)
        else:
            print('model/model.params does not exist and will be created.')

    # Get our data loaders.
    train_data_loader = get_data_loader(
        train_dataset, transformer=basic_transform, batch_size=args.batch_size, shuffle=True
    )

    test_data_loader = get_data_loader(
        test_dataset, transformer=basic_transform, batch_size=args.batch_size, shuffle=False
    )

    # Don't use a dataloader for the test data. It's easier not to use batches here.
    test_dataset = gluon.data.ArrayDataset(test_dataset).transform(basic_transform)

    if not args.no_train:
        train_model(model=model,
                    train_data_loader=train_data_loader, test_dataset=test_data_loader,
                    num_epochs=args.epochs)

    predict_on_test(
        model=model, test_dataset=test_data_loader,
        label_map=basic_transform.get_label_map()
    )
