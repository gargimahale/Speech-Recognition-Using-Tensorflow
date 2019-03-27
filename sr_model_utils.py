import numpy as np
import tensorflow as tf

def minibatches(inputs, targets=None, minibatch_size=64):
    """batch generator. yields x and y batch.
    """
    x_batch, y_batch = [], []

    if targets is not None:
        for inp, tgt in zip(inputs, targets):
            if len(x_batch) == minibatch_size and len(y_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            x_batch.append(inp)
            y_batch.append(tgt)

        if len(x_batch) != 0:
            for inp, tgt in zip(inputs, targets):
                if len(x_batch) != minibatch_size:
                    x_batch.append(inp)
                    y_batch.append(tgt)
                else:
                    break
            yield x_batch, y_batch
    else:
        for inp in inputs:
            if len(x_batch) == minibatch_size:
                yield x_batch
                x_batch = []
            x_batch.append(inp)

        if len(x_batch) != 0:
            for inp in inputs:
                if len(x_batch) != minibatch_size:
                    x_batch.append(inp)
                else:
                    break
            yield x_batch


def pad_txt_sequences(sequences, pad_tok):
    """Pads the sentences, so that all sentences in a batch have the same length.
    """

    max_length = max(len(x) for x in sequences)

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq + [pad_tok] * max(max_length - len(seq), 0)

        sequence_padded += [seq_]
        sequence_length += [len(seq)]

    return sequence_padded, sequence_length

def pad_audio_sequences(sequences, tail=True):
    """

    Args:
        sequences: Array of audio sequences
        tail: Boolean. Append silence to end or beginning

    Returns: Padded array with audio sequences, padded with
             silence.

    """

    max_length = max(seq.shape[0] for seq in sequences)

    sequences_padded, sequence_length = [], []

    for seq in sequences:
        if tail:
            seq_shape = seq.shape
            pad_vector = [0] * seq_shape[1]
            n_vectors_to_add = max_length - seq_shape[0]

            for _ in range(n_vectors_to_add):
                seq = np.append(seq, [pad_vector], axis=0)

        sequences_padded.append(seq)
        sequence_length.append(seq_shape[0])


    return sequences_padded, sequence_length


def reset_graph(seed=97):
    """helper function to reset the default graph. this often
       comes handy when using jupyter noteboooks.
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
