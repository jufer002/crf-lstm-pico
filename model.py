import mxnet as mx

from mxnet import gluon
from mxnet.gluon.block import Block
from crf.crf import CRF
from lstm_crf.lstm_crf import BiLSTM_CRF


class PicoExtractor(Block):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    """

    def __init__(self, tag2Idx, ctx, vocab_size, embedding_dim, lstm_hidden_dim):
        super(PicoExtractor, self).__init__()

        self.num_labels = len(tag2Idx)

        # Model layers defined here.
        with self.name_scope():
            self.lstm_crf = BiLSTM_CRF(
                vocab_size=vocab_size,
                tag2Idx=tag2Idx,
                embedding_dim=embedding_dim,
                hidden_dim=lstm_hidden_dim
            )
            self.output = gluon.nn.Sequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(self.num_labels))


    def forward(self, data):
        score, tag_seq = lstm_output = self.lstm_crf(data)

        output = self.output(lstm_output)
        return output
