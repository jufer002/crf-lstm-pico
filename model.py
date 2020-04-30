from mxnet.gluon.block import Block
from crf.crf import CRF


class PicoExtractor(Block):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    """

    def __init__(self, tag2idx, ctx):
        super(PicoExtractor, self).__init__()

        # Model layers defined here.
        with self.name_scope():
            self.crf = CRF(tag2idx=tag2idx, ctx=ctx)

    def forward(self, f, data):
        pass
