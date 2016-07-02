
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Flatten, Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import merge
from keras.models import Model
from keras.layers.recurrent import LSTM


def build_model(vocab_size, embed_dim=50, level='word', token_len=15,
                token_char_vector_dict={}, nb_recurrent_layers=3):
    if level == 'word':
        pivot_inp = Input(shape=(1, ), dtype='int32', name='pivot')
        pivot_embed = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim)(pivot_inp)

    elif level == 'char':
        pivot_inp = Input(shape=(token_len, len(token_char_vector_dict)),
                          name='pivot')
        for i in range(nb_recurrent_layers):
            if i == 0:
                curr_input = pivot_inp
            else:
                curr_input = curr_out

            l2r = LSTM(output_dim=embed_dim,
                       return_sequences=True,
                       activation='tanh')(curr_input)
            r2l = LSTM(output_dim=embed_dim,
                       return_sequences=True,
                       activation='tanh',
                       go_backwards=True)(curr_input)
            curr_out = merge([l2r, r2l], name='encoder_'+str(i+1), mode='sum')

        flattened = Flatten()(curr_out)
        pivot_embed = Dense(embed_dim)(flattened)
        pivot_embed = Reshape((1, embed_dim))(pivot_embed)

    context_inp = Input(shape=(1, ), dtype='int32', name='context')
    context_embed = Embedding(
        input_dim=vocab_size, output_dim=embed_dim)(context_inp)

    prod = merge([pivot_embed, context_embed], mode='dot', dot_axes=2)
    res = Reshape((1, ), input_shape=(1, 1))(prod)

    activ = Activation('sigmoid', name='label')(res)

    model = Model(input=[pivot_inp, context_inp], output=activ)

    optim = RMSprop()
    model.compile(loss='mse', optimizer=optim)
    return model
