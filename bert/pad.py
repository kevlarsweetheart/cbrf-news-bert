from torch.nn.utils.rnn import pad_sequence as pad_sequence_


def pad_sequence(sequences, fill=(0, 0)):
    max_len = len(max(sequences, key=lambda _: len(_)))
    for seq in sequences:
        pad_len = max_len - len(seq)
        padding = [fill] * pad_len
        seq += padding
    return sequences


def pad_bool_sequence(sequences, fill=False):
    return pad_sequence_(
        sequences,
        batch_first=True,
        padding_value=fill
    )
