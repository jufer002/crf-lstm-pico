"""
Author: Julian Fernandez

This module is a thin layer on top of the lstm_crf module, which is found here:
https://github.com/apache/incubator-mxnet/tree/master/example/gluon/lstm_crf
"""

import mxnet as mx

from concurrent.futures import ThreadPoolExecutor
from mxnet.gluon.block import Block
from lstm_crf.lstm_crf import BiLSTM_CRF


class PicoExtractor(Block):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    """

    def __init__(self, tag2Idx, ctx, vocab_size, embedding_dim, lstm_hidden_dim, dropout):
        super(PicoExtractor, self).__init__()

        self.ctx = ctx
        self.num_labels = len(tag2Idx)
        self.label_map = tag2Idx

        # Model layers defined here.
        with self.name_scope():
            self.lstm_crf = BiLSTM_CRF(
                vocab_size=vocab_size,
                tag2Idx=tag2Idx,
                embedding_dim=embedding_dim,
                hidden_dim=lstm_hidden_dim,
                ctx=ctx,
                dropout=dropout
            )

    def neg_log_likelihood(self, data, labels, ag=None):
        # Slice data like this because the lstm_crf._score_sentence function
        # does not expect <bos> and <eos> tokens in the data; only in the
        # tags.
        data = mx.nd.array(data[:, 1:-1])
        # If no autograd object is passed in, loop through the batch and feed it one-by-one.
        if ag is None:
            log_like = mx.nd.array([0], ctx=self.ctx)
            for (x, y) in zip(data, labels):
                log_like = log_like + self.lstm_crf.neg_log_likelihood(x, y)

            return log_like

        # Otherwise, try an experimental multithreaded approach.
        funcs = []
        for (x, y) in zip(data, labels):
            def compute_neg_log_likelihood():
                with ag.record():
                    return self.lstm_crf.neg_log_likelihood(x, y)

            funcs.append(compute_neg_log_likelihood)

        log_likes = []
        with ThreadPoolExecutor() as executor:
            for func in funcs:
                call = executor.submit(func)
                log_likes.append(call.result())

        log_like = mx.nd.array([0], ctx=self.ctx)
        for loss in log_likes:
            log_like = log_like + loss

        return log_like

    # Manually override initialize function to make sure lstm_crf initializes properly.
    def initialize(self, init=mx.initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False, vocabulary=None):
        super().initialize(init, ctx, verbose, force_reinit)
        self.lstm_crf.initialize(init, ctx, verbose, force_reinit)
        self.lstm_crf.word_embeds.weight.set_data(vocabulary.embedding.idx_to_vec)

    def forward(self, data):
        data = mx.nd.array(data[1:-1])
        # Loop through each sentence
        score, tag_seq = self.lstm_crf(data)

        # output = self.output(tag_seq)
        return score, tag_seq
