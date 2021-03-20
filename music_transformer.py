"""
Copyright 2020 Aditya Gomatam.

This file is part of Music-Transformer (https://github.com/spectraldoy/Music-Transformer), my project to build and
train a Music Transformer. Music-Transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. Music-Transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. Music-Transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for Music-Transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""

import tensorflow as tf
import numpy as np

import os
import math

from IPython.display import Audio

import transformerutil6 as tu

MAX_LENGTH = 1921

"""Absolute Position Encoding"""


def get_angles(position, k, d_model):
    # all values of each k
    angle = 1 / np.power(10000, 2 * (k // 2) / d_model)
    # matrix multiplied into all positions - represent each position with a d_model sized vector
    return position @ angle


def abs_positional_encoding(max_position, d_model, n=3):
    """
    returns absolute position encoding, creating a vector representation for all positions
    from 0 to max_position of shape (d_model,) -> a matrix of shape (max_position, d_model)
    and broadcasts it to n dimensions
    """
    # angles are of shape (positions, d_model)
    angles = get_angles(np.arange(max_position)[:, np.newaxis],
                        np.arange(d_model)[np.newaxis, :],
                        d_model)

    # apply sin to the even indices along the last axis
    angles[:, 0::2] = np.sin(angles[:, 0::2])

    # apply cos to the odd indices along the last axis
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    # broadcast to n dimensions
    for _ in range(n - 2):
        angles = angles[np.newaxis, :]
    return tf.cast(angles, tf.float32)


"""Masking"""


def create_padding_mask(seq, n=4):
    """
    Creates padding mask for a batch of sequences seq. Mask will be of shape
    (batch_size, seq_len), and can be broadcasted to n dimensions
    """
    mask = tf.cast(tf.equal(seq, 0), tf.float32)  # mask is 1 where seq is 0
    # reshape to # batch_size, 1, ..., 1. seq_len
    return tf.reshape(mask, (tf.shape(mask)[0], *[1 for _ in range(n - 2)], tf.shape(mask)[-1]))


def create_look_ahead_mask(seq_len):
    """
    Creates an upper triangular mask of ones of shape (seq_len seq_len).
    It is the same for all inputs of shape seq_len
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask, tf.float32)  # (seq_len, seq_len)


def create_mask(inp, n=4):
    """
    function to create the proper mask for an input batch
    mask = max(padding_mask, look_ahead_mask)

    Args:
      inp: batch tensor of input sequences of shape (..., seq_len)
    """
    padding_mask = create_padding_mask(inp, n)
    look_ahead_mask = create_look_ahead_mask(inp.shape[-1])

    # create final mask
    return tf.maximum(padding_mask, look_ahead_mask)


"""Self-Attention with Relative Position Embeddings"""


def skew(t: tf.Tensor):
    """
    Implements skewing algorithm given by Huang et. al 2018 to reorder the
    dot(Q, RelativePositionEmbeddings) matrix into the correct ordering for which
    Tij = compatibility of ith query in Q with relative position (j - i)

    This implementation accounts for rank n tensors

    Algorithm:
        1. Pad T
        2. Reshape
        3. Slice

    T is supposed to be of shape (..., L, L), but the function generalizes to any shape
    """
    # pad the input tensor
    middle_paddings = [[0, 0] for _ in range(len(t.shape) - 1)]
    padded = tf.pad(t, [*middle_paddings, [1, 0]])

    # reshape
    Srel = tf.reshape(padded, (-1, t.shape[-1] + 1, t.shape[-2]))
    Srel = Srel[:, 1:]  # slice required positions
    return tf.cast(tf.reshape(Srel, t.shape), t.dtype)


def rel_scaled_dot_prod_attention(q, k, v, e, mask=None):
    """
    Implements equation 3 given in the previous section to calculate the attention weights,
    Mask has different shapes depending on its type (padding, look_ahead or combined),
    but by scaling and adding it to the attention logits, masking can be performed

    Attention = softmax(mask(QKT + skew(QET))/sqrt(d_k))V

    Args:
      q: Queries matrix of shape (..., seq_len_q, d_model)
      k: Keys matrix of shape (..., seq_len_k, d_model)
      v: Values matrix of shape (..., seq_len_k, d_model)
      e: Relative Position embedding matrix of shape (seq_len_k, d_model)

    Returns:
      output attention, attention weights
    """
    QKt = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    Srel = skew(tf.matmul(q, e, transpose_b=True))  # (..., seq_len_q, seq_len_k)

    # calculate and scale logits
    dk = math.sqrt(k.shape[-1])
    scaled_attention_logits = (QKt + Srel) / dk

    # add the mask to the attention logits
    if mask is not None:
        scaled_attention_logits += (mask * -1e09)  # mask is added only to attention logits

    # softmax is normalized on the last axis so that the ith row adds up to 1
    # this is best for multiplication by v because the last axis (made into
    # probabilities) interacts with the values v
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, d_k)
    return output, attention_weights


# Multi-Head Attention
def split_heads(x, num_heads, depth=None):
    """
    assumes x is of shape (..., num_heads * depth)
    split the last dimension of x into (num_heads, depth),
    transposes to (..., num_heads, L, depth)
    """
    if depth is None:
        assert x.shape[-1] % num_heads == 0
        depth = x.shape[-1] // num_heads

    # split d_model into h, d_h
    x = tf.reshape(x, (*x.shape[:-1], num_heads, depth))  # (..., L, num_heads, depth)

    # transpose axes -2 and -3 - tf specifies this with perm so all this fluff needs to be done
    final_perm = len(x.shape) - 1
    prior_perms = np.arange(0, final_perm - 2)  # axes before the ones that need to be transposed

    # transpose to shape (..., num_heads, L, depth)
    return tf.transpose(x, perm=[*prior_perms, final_perm - 1, final_perm - 2, final_perm])


# another helper function
def get_required_embeddings(E, seq_len, max_len=None):
    """
    Given an input sequence of length seq_len, which does not necessary equal max_len, the
    maximum relative distance the model is set to handle, embeddings in E from the right are
    the required relative positional embeddings
    Embeddings have to be taken from the right because E is considered to be
    ordered from -max_len + 1 to 0
    For all positions distanced past -max_len + 1, use E_{-max_len + 1}
    """
    if not E.built:
        E.build(seq_len)
    if max_len is None:
        max_len = E.embeddings.get_shape()[0]  # assumes E is a keras.layers.Embedding

    if max_len >= seq_len:
        seq_len = min(seq_len, max_len)
        return E(np.arange(max_len - seq_len, max_len))

    return tf.concat(
        values=[*[E(np.arange(0, 1)) for _ in range(seq_len - max_len)], E(np.arange(0, max_len))],
        axis=0
    )


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_rel_dist=MAX_LENGTH, use_bias=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_len = max_rel_dist

        assert d_model % num_heads == 0, "d_model must be divisible into num_heads"

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=use_bias)  # parameter matrix to generate Q from input
        self.wk = tf.keras.layers.Dense(d_model, use_bias=use_bias)  # parameter matrix to generate K from input
        self.wv = tf.keras.layers.Dense(d_model, use_bias=use_bias)  # parameter matrix to generate V from input

        self.E = tf.keras.layers.Embedding(self.max_len, self.d_model)  # relative position embeddings

        self.wo = tf.keras.layers.Dense(d_model, use_bias=use_bias)  # final output parameter matrix

    def call(self, q, k, v, mask=None):
        """
        Creates Q, K, and V, gets required embeddings in E, splits into heads,
        computes attention, concatenates, and passes through final output layer
        """
        # Get Q, K, V
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # Get E
        seq_len_k = k.shape[-2]
        e = get_required_embeddings(self.E, seq_len_k, self.max_len)  # (seq_len_k, d_model)

        # split into heads
        q = split_heads(q, self.num_heads, self.depth)  # (batch_size, h, seq_len_q, depth)
        k = split_heads(k, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        v = split_heads(v, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        e = split_heads(e, self.num_heads, self.depth)  # (            h, seq_len_k, depth)

        # rel_scaled_attention shape = (batch_size, h, seq_len_q, depth)
        # attention_weights shape = (batch_size, h, seq_len_q, seq_len_k)
        rel_scaled_attention, attention_weights = rel_scaled_dot_prod_attention(q, k, v, e, mask=mask)

        # transpose rel_scaled_attention back to (batch_size seq_len_q, h, depth)
        final_perm = len(rel_scaled_attention.shape) - 1  # can't use rank for some reason
        prior_perms = np.arange(0, final_perm - 2)  # axes before the ones that need to be transposed
        rel_scaled_attention = tf.transpose(rel_scaled_attention,
                                            perm=[*prior_perms, final_perm - 1, final_perm - 2, final_perm])

        # concatenate heads -> (batch_size, seq_len, d_model)
        sh = rel_scaled_attention.shape
        concat_attention = tf.reshape(rel_scaled_attention, (*sh[:-2], self.d_model))

        output = self.wo(concat_attention)

        return output, attention_weights


"""Pointwise FFN"""


class PointwiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, use_bias=True):
        super(PointwiseFFN, self).__init__()

        self.main = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', use_bias=use_bias),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model, use_bias=use_bias)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x):
        return self.main(x)


"""Decoder Layer"""


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, max_rel_dist=MAX_LENGTH,
                 use_bias=True, dropout=0.1, layernorm_eps=1e-06):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, max_rel_dist=max_rel_dist, use_bias=use_bias)
        self.ffn = PointwiseFFN(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=layernorm_eps)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_output, attn_weights = self.mha(x, x, x, mask=mask)  # calculate attention
        attn_output = self.dropout1(attn_output, training=training)  # dropout
        # layernorm on residual connection
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)

        ffn_output = self.ffn(out1)  # pass through FFN
        ffn_output = self.dropout2(ffn_output, training=training)  # dropout
        # layernorm on residual connection
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)

        return out2, attn_weights


"""Music Transformer Decoder"""


class MusicTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_rel_dist=MAX_LENGTH,
                 max_abs_position=20000, use_bias=True, dropout=0.1, layernorm_eps=1e-06, tie_emb=False):
        super(MusicTransformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.tie_emb = tie_emb

        self.max_position = max_abs_position  # might need for decode

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)  # input embeddings
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)  # absolute position encoding
        self.dropout = tf.keras.layers.Dropout(dropout)  # embedding dropout

        # decoder layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, max_rel_dist, use_bias, dropout, layernorm_eps) \
                           for _ in range(self.num_layers)]

        # final layer is linear or embedding weight depending on tie emb
        if not tie_emb:
            self.final_layer = tf.keras.layers.Dense(vocab_size, use_bias=use_bias)

    def call(self, x, training=False, mask=None):
        # initialize attention weights dict to output
        attention_weights = {}

        # embed x and add absolute positional encoding
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x += self.positional_encoding[:, :x.shape[-2], :]

        x = self.dropout(x, training=training)

        # pass through decoder layers
        for i in range(len(self.dec_layers)):
            x, w_attn = self.dec_layers[i](x, training, mask)
            attention_weights[f'DecoderLayer{i + 1}'] = w_attn

        # final layer
        if self.tie_emb:
            x = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
        else:
            x = self.final_layer(x)

        # returns unsoftmaxed logits
        return x, attention_weights


"""Hyperparameters"""

num_layers = 6
d_model = 256
dff = 1024
num_heads = 8

use_bias = True
tie_emb = False
layernorm_eps = 1e-06

vocab_size = tu.vocab_size
dropout_rate = 0.1

hparams = (num_layers, d_model, num_heads, dff, vocab_size, MAX_LENGTH,
           use_bias, dropout_rate, layernorm_eps, tie_emb)

"""Creating the Model

# instantiate
transformer = Music-Transformer(*hparams)

# build
_ = transformer(tf.random.uniform((2, MAX_LENGTH)))
del _

# transformer.load_weights(PATH)
"""

"""Generate!"""


def greedy_decode(transformer, inp, mode='categorical', temperature=1.0, k=None, skip_ends=0, memory=1000):
    """
    Decodes inp greedily by appending last outputs to the input and feeding
    back into the model. Model is made to generate until end token is predicted
    by feeding only the last model.max_len inputs to the model at each decode step
    """
    # get tokens
    if not isinstance(inp, tf.Tensor) and not isinstance(inp, np.ndarray):
        inp = tu.events_to_indices(inp)
    if inp[0] != tu.start_token:
        middle_dims = [[0, 0] for _ in range(tf.rank(inp) - 1)]
        inp = tf.pad(inp, paddings=[*middle_dims, [1, 0]], constant_values=tu.start_token)
    # check if temperature / k is a function
    if not callable(temperature):
        temperature_ = temperature;
        del temperature
        temperature = lambda x: temperature_

    if not callable(k) and k is not None:
        k_ = k
        del k
        k = lambda x: k_

    # dimension for the mask
    n = tf.rank(inp) + 2 if tf.rank(inp) > 0 else 3

    # make inp 2d
    inp = [tf.expand_dims(inp, 0)]

    # initialize attention weights in case inp.shape[-1] is already > max_len
    attention_weights = {}

    # maximum number of tokens to input to the model
    try:
        while True:
            predictions, attention_weights = transformer(inp[-1], training=False,
                                                         mask=create_mask(inp[-1], n))

            # divide logits by temperature
            predictions /= temperature(inp[-1].shape[-1])

            # get last prediction
            if mode == 'argmax' or mode == 'a':
                prediction = tf.expand_dims(tf.argmax(predictions[..., -1, :], axis=-1, output_type=tf.int32), 0)
            elif k is not None:
                top_k_final_predictions = tf.math.top_k(predictions[..., -1, :],
                                                        k=k(inp[-1].shape[-1]))
                predicted_idx = tf.random.categorical(
                    logits=top_k_final_predictions.values,
                    num_samples=1,
                    dtype=tf.int32
                )
                predicted_idx = tf.squeeze(predicted_idx)
                prediction = tf.expand_dims(tf.expand_dims(top_k_final_predictions.indices[0, predicted_idx], 0), 0)
            elif mode == 'categorical' or mode == 'c':
                prediction = tf.random.categorical(logits=predictions[..., -1, :], num_samples=1, dtype=tf.int32)
            else:
                print(f"Unsupported mode '{mode}'. Use 'argmax' or 'categorical'")
                return None

            # return if prediction is end token
            if prediction == tu.end_token:
                if skip_ends <= 0:
                    out = tf.concat(inp, axis=-1)
                    return tf.squeeze(out)[1:], attention_weights
                else:
                    skip_ends -= 1
                    vec = inp[-1]
                    inp.append(vec[:, :-memory])
                    # maybe i need to put the start token here so that it actually ends at 1920 positions
                    inp.append(vec[:, -memory:])
                    inp = inp[:-3] + inp[-2:]
            # else concatenate last output to inp
            inp[-1] = tf.concat([inp[-1], prediction], axis=-1)
    except KeyboardInterrupt:
        pass
    out = tf.concat(inp, axis=-1)
    return tf.squeeze(out)[1:], attention_weights


def audiate(idx_list, path='./bloop.mid', tempo=512820, gain=1.0, sr=44100, wav=True, verbose=False):
    # check path is mid or midi, set to mid, else invalid path
    if path.endswith("midi"):
        path = path[:-1]
    elif path.endswith("mid"):
        pass
    else:
        print("Invalid extension. Use '.mid' or '.midi'.")
        return None

        # create and save the midi file
    print("Saving midi file...") if verbose else None
    mid = tu.Listparser(index_list=idx_list, tempo=tempo)
    mid.save(path)
    if not wav:
        print(f"Midi saved at {path}")
        return None
    # run the FluidSynth command
    print("Creating wav file...\n") if verbose else None
    os.system(f"fluidsynth -ni Yamaha-C5-Salamander-JNv5.1.sf2 {path} -F {path[:-4]}.wav -r 44100 -g {gain}")

    return Audio(f"{path[:-4]}.wav")


def generate(transformer, inp, path='./bloop.mid', mode='categorical', temperature=1.0,
             k=None, skip_ends=0, memory=1000, tempo=512820, wav=True, verbose=False):
    # get the index list
    if verbose:
        print("Greedy decoding...", end='')
        start = time.time()
        idx_list, attn_weights = greedy_decode(transformer, inp, mode, temperature,
                                               k, skip_ends, memory)
        end = time.time()
        print(f"Generated {len(idx_list)} tokens.", end=" ")
        print(f"Time taken: {round(end - start, 2)} secs.")
    else:
        idx_list, attn_weights = greedy_decode(transformer, inp, mode, temperature,
                                               k, skip_ends, memory)

    # generate audio
    return audiate(idx_list, path, tempo, wav=wav, verbose=verbose)
