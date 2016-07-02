
import numpy as np


def index_characters(tokens):
    vocab = {ch for tok in tokens for ch in tok.lower()}

    vocab = vocab.union({'$', '|', '%'})

    char_vocab = tuple(sorted(vocab))
    char_vector_dict, char_idx = {}, {}
    filler = np.zeros(len(char_vocab), dtype='float32')

    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
        char_idx[idx] = char

    return char_vector_dict, char_idx


def vectorize_token(seq, char_vector_dict, max_len):
    # cut, if needed:
    seq = seq[:(max_len - 2)]
    seq = '%' + seq + '|'
    seq = seq[::-1]  # reverse order (cf. paper)!

    filler = np.zeros(len(char_vector_dict), dtype='float32')

    seq_X = []
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)

    while len(seq_X) < max_len:
        seq_X.append(filler)

    return np.array(seq_X, dtype='float32')


def vectorize_tokens(tokens, char_vector_dict, max_len=15):
    X = []
    for token in tokens:
        token = token.lower()
        x = vectorize_token(seq=token,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    return np.asarray(X, dtype='float32')
