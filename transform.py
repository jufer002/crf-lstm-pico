import mxnet as mx


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 250
        Maximum sequence length - longer seqs will be truncated and shorter ones padded

    """
    def __init__(self, labels, max_len=250):
        self._max_seq_length = max_len
        self._label_map = {
            # Use i+1 to reserve 0 for the pad-label.
            label: i+1
            for (i, label) in enumerate(labels)
        }

    def __call__(self, data, labels):
        label_ids = [self._label_map[label] for label in labels]

        # Pad labels if necessary.
        padded_label_ids = label_ids + [0] * (self._max_seq_length - len(label_ids))
        # Pad data.
        padded_data = data + [0] * (self._max_seq_length - len(data))

        data_array = mx.nd.array(padded_data, dtype='int32')
        label_array = mx.nd.array(padded_label_ids, dtype='int32')

        return data_array, label_array

    def get_label_map(self):
        return self._label_map

