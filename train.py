import argparse
import gluonnlp as nlp
import mxnet as mx

from load_data import load_dataset
from model import PicoExtractor
from mxnet import autograd
from mxnet import gluon


def get_data_loader(dataset, transformer, batch_size, shuffle):
    transformed_dataset = gluon.data.ArrayDataset(dataset).transform(transformer)

    return gluon.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(label_map, ctx):
    pico_model = PicoExtractor(tag2idx=label_map, ctx=ctx)
    pico_model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)

    return pico_model


def train_model(model, train_data_loader, test_data_loader, num_epochs):
    differentiable_params = []

    # Do not apply weight decay on LayerNorm and bias terms.
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    # Train model.
    for epoch in range(num_epochs):
        epoch_loss = 0

        for i, (data_out_of_context, label_out_of_context) in enumerate(train_data_loader):
            # Contextualize data and the label.
            data = data_out_of_context.as_in_context(ctx)
            label = label_out_of_context.as_in_context(ctx)

            out = model(data)

        with autograd.record():
            pass

        trainer.step(1)


if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
    parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data',
                        default='data/train.jsonl')
    parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data',
                        default='data/test.jsonl')
    parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
    parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.3)
    parser.add_argument('--embedding_source', type=str, default='glove.6B.100d',
                        help='Pre-trained embedding source name')

    args = parser.parse_args()

    # Load the vocabulary, dataset, and data transformer.
    vocabulary, train_dataset, test_dataset, basic_transform = load_dataset(
        # TODO Use higher max length
        train_file=args.train_file, test_file=args.test_file, max_length=32
    )

    # Set the word embeddings.
    if args.embedding_source:
        glove_embeddings = nlp.embedding.create('glove', source=args.embedding_source)

        vocabulary.set_embedding(glove_embeddings)

    # Create context object.
    ctx = mx.cpu()

    # Build a model.
    model = build_model(basic_transform.get_label_map(), ctx=ctx)

    # Get our data loaders.
    train_data_loader = get_data_loader(
        train_dataset, transformer=basic_transform, batch_size=args.batch_size, shuffle=True
    )

    test_data_loader = get_data_loader(
        test_dataset, transformer=basic_transform, batch_size=args.batch_size, shuffle=False
    )

    train_model(model=model,
                train_data_loader=train_data_loader, test_data_loader=test_data_loader,
                num_epochs=args.epochs)
