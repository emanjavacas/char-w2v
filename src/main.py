from __future__ import print_function

import codecs
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from keras.utils import np_utils
from keras import backend as K

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from utils import index_characters, vectorize_tokens
from word2vec import make_sampling_table, skipgrams
from model import build_model


def main():
    MAX_VOCAB = 6000
    WINDOW_SIZE = 4
    LEVEL = 'char'
    EMBED_DIM = 100
    MAX_TOKEN_LEN = 15
    NB_LAYERS = 1
    NB_EPOCHS = 3

    cutoff = 10000000
    words = codecs.open('../data/Austen_Sense.txt', 'r', encoding='utf8') \
                  .read().lower().split()[:cutoff]
    print('Loaded', len(words), 'words')

    cnt = Counter(words)
    most_comm = [k for k, v in cnt.most_common(500)]
    print('Most frequent:', most_comm[:50])

    word_to_int = {'UNK': 0}
    for w, c in cnt.most_common(MAX_VOCAB):
        word_to_int[w] = len(word_to_int)
    int_to_word = [None] * len(word_to_int)
    for k, v in word_to_int.items():
        int_to_word[v] = k

    if LEVEL == 'char':
        char_vector_dict, char_idx = index_characters(int_to_word)
        print(char_vector_dict.keys())
        model = build_model(vocab_size=len(word_to_int),
                            embed_dim=EMBED_DIM,
                            level=LEVEL,
                            token_len=MAX_TOKEN_LEN,
                            token_char_vector_dict=char_vector_dict,
                            nb_recurrent_layers=NB_LAYERS)

        most_comm_X = vectorize_tokens(tokens=most_comm,
                                       char_vector_dict=char_vector_dict,
                                       max_len=MAX_TOKEN_LEN)
        print(most_comm_X.shape, '!!!')

    elif LEVEL == 'word':
        model = build_model(vocab_size=len(word_to_int),
                            embed_dim=50,
                            level=LEVEL,
                            token_len=None,
                            token_char_vector_dict=None,
                            nb_recurrent_layers=None)
    model.summary()

    sampling_table = make_sampling_table(size=len(word_to_int))

    for e in range(NB_EPOCHS):
        idx = 0
        losses = []

        for idx in range(WINDOW_SIZE, len(words)-WINDOW_SIZE):
            seq = []
            for w in words[(idx - WINDOW_SIZE): (idx + WINDOW_SIZE)]:
                try:
                    seq.append(word_to_int[w])
                except KeyError:
                    seq.append(0)

            couples, labels = skipgrams(seq, len(word_to_int),
                                        window_size=4,
                                        negative_samples=1.,
                                        shuffle=True,
                                        categorical=False,
                                        sampling_table=sampling_table)

            if len(couples) > 1:
                couples = np.array(couples, dtype='int32')

                c_inp = couples[:, 1]
                c_inp = c_inp[:, np.newaxis]

                if LEVEL == 'word':
                    p_inp = couples[:, 0]
                    p_inp = p_inp[:, np.newaxis]
                elif LEVEL == 'char':
                    tokens = [int_to_word[i] for i in couples[:, 0]]
                    p_inp = vectorize_tokens(tokens=tokens,
                                             char_vector_dict=char_vector_dict,
                                             max_len=MAX_TOKEN_LEN)
                else:
                    raise ValueError('Wrong level param: word or char')

                labels = np.array(labels, dtype='int32')
                
                loss = model.train_on_batch({'pivot': p_inp, 'context': c_inp},
                                            {'label': labels})
                losses.append(loss)

                if idx % 5000 == 0:
                    print(np.mean(losses))

                if idx % 10000 == 0:
                    print(np.mean(losses))

                    print('Compiling repr func')
                    get_activations = K.function([model.layers[0].input,
                                                  K.learning_phase()],
                                                 [model.layers[6].output, ])
                    activations = get_activations([most_comm_X, 0])[0]
                    activations = np.array(activations, dtype='float32')

                    print(activations.shape, '-----')
                    norm_weights = np_utils.normalize(activations)

                    # dimension reduction:
                    tsne = TSNE(n_components=2)
                    coor = tsne.fit_transform(norm_weights)

                    plt.clf()
                    sns.set_style('dark')
                    sns.plt.rcParams['axes.linewidth'] = 0.4
                    fig, ax1 = sns.plt.subplots()

                    labels = most_comm
                    # first plot slices:
                    x1, x2 = coor[:, 0], coor[:, 1]
                    ax1.scatter(x1, x2, 100,
                                edgecolors='none',
                                facecolors='none')
                    # clustering on top (add some colouring):
                    clustering = AgglomerativeClustering(linkage='ward',
                                                         affinity='euclidean',
                                                         n_clusters=10)
                    clustering.fit(coor)
                    # add names:
                    axes = zip(x1, x2, most_comm, clustering.labels_)
                    for x, y, name, cluster_label in axes:
                        ax1.text(x, y, name, ha='center', va="center",
                                 color=plt.cm.spectral(cluster_label / 10.),
                                 fontdict={'family': 'Arial', 'size': 8})
                    # control aesthetics:
                    ax1.set_xlabel('')
                    ax1.set_ylabel('')
                    ax1.set_xticklabels([])
                    ax1.set_xticks([])
                    ax1.set_yticklabels([])
                    ax1.set_yticks([])
                    sns.plt.savefig('embeddings.pdf', bbox_inches=0)

"""
# recover the embedding weights trained with skipgram:
weights = model.layers[0].get_weights()[0]

# we no longer need this
del model

weights[:skip_top] = np.zeros((skip_top, dim_proj))
norm_weights = np_utils.normalize(weights)

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])
word_index = tokenizer.word_index

def embed_word(w):
    i = word_index.get(w)
    if (not i) or (i<skip_top) or (i>=max_features):
        return None
    return norm_weights[i]

def closest_to_point(point, nb_closest=10):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]  

def closest_to_word(w, nb_closest=10):
    i = word_index.get(w)
    if (not i) or (i<skip_top) or (i>=max_features):
        return []
    return closest_to_point(norm_weights[i].T, nb_closest)


''' the resuls in comments below were for: 
    5.8M HN comments
    dim_proj = 256
    nb_epoch = 2
    optimizer = rmsprop
    loss = mse
    max_features = 50000
    skip_top = 100
    negative_samples = 1.
    window_size = 4
    and frequency subsampling of factor 10e-5. 
'''

words = ["article", # post, story, hn, read, comments
"3", # 6, 4, 5, 2
"two", # three, few, several, each
"great", # love, nice, working, looking
"data", # information, memory, database
"money", # company, pay, customers, spend
"years", # ago, year, months, hours, week, days
"android", # ios, release, os, mobile, beta
"javascript", # js, css, compiler, library, jquery, ruby
"look", # looks, looking
"business", # industry, professional, customers
"company", # companies, startup, founders, startups
"after", # before, once, until
"own", # personal, our, having
"us", # united, country, american, tech, diversity, usa, china, sv
"using", # javascript, js, tools (lol)
"here", # hn, post, comments
]

for w in words:
    res = closest_to_word(w)
    print('====', w)
    for r in res:
        print(r)
"""

if __name__ == '__main__':
    main()
