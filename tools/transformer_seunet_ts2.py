import gc
import os
import random

import h5py
import numpy as np
from keras.layers import Dense, Conv1D, ELU,Flatten, Dropout, BatchNormalization, MaxPooling1D, Add
from keras.models import Model
from keras.layers import Input

from keras.callbacks import ModelCheckpoint
from keras.optimizer_v2.adam import Adam
# from keras.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import tensorflow as tf
from tensorflow import keras


# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

def get_positional_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    position_embedding = np.zeros((sentence_length, d_model))
    position_embedding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    position_embedding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    #    sines = np.sin(angle_rads[:, 0::2])
    #    cosines = np.cos(angle_rads[:, 1::2])
    position_embedding = position_embedding[np.newaxis, ...]
    return tf.cast(position_embedding, dtype = tf.float32) # 类型转换函数 cast

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, ((2 * i) / np.float32(d_model)))
    return pos * angle_rates


def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    - q: shape == (..., seq_len_q, depth)
    - k: shape == (..., seq_len_k, depth)
    - v: shape == (..., seq_len_v, depth_v)
    - seq_len_k == seq_len_v
    - mask: shape == (..., seq_len_q, seq_len_k) # 对q * k 之后的矩阵进行mask
    Return:
    - output: weighted sum
    - attention_weights: weight of attention
    """

    # matmul_qk.shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # transpose_b表示第二个（k）做转置

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logit = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # 为使得在 softmax 后值趋近于0
        scaled_attention_logit += (mask * -1e9)

    # attention_weights.shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logit, axis=-1)

    # output.shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

# def print_scaled_dot_product_attention(q, k, v):
#     temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
#     print("Attention weights are:")
#     print(temp_att)
#     print("Output is:")
#     print(temp_out)


class MultiHeadAttention(keras.layers.Layer):
    """
    理论上：
    x -> wq0 -> q0
    x -> wk0 -> k0
    x -> wv0 -> v0

    实战中：
    q -> wq0 -> q0
    x -> wk0 -> k0
    x -> wv0 -> v0

    实战中的技巧：
    q -> wq -> Q -> split -> q0, q1, q2...
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)




        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # 多头： x -> (batch_size, num_heads, seq_len, depth)

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 维度重排列
        # return (batch_size, num_heads, seq_len, depth)

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)  # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k)  # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v)  # v.shape: (batch_size, seq_len_v, d_model)

        # q.shape: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # k.shape: (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # v.shape: (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = \
            scaled_dot_product_attention(q, k, v, mask)

        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])  # 把多头合并起来
        concat_attention = tf.reshape(scaled_attention_outputs, (batch_size, -1, self.d_model))

        # output.shape: (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights



# Feedforwared
def feed_forward_network(d_model, dff):
    # dff: dim of forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])


def pre_net(d_model):
    return keras.layers.Dense(d_model)


# Encode Layers
class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normlize & dropout
      -> feed_forward -> add & normlize
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        
        self.add1 = keras.layers.Add()
        self.add2 = keras.layers.Add()

    def call(self, x, training, encoder_padding_mask):
        # x.shape          :(batch_size, seq_leb, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
        
        ln_x = self.layer_norm1(x)
        attn_output, _ = self.mha(ln_x, ln_x, ln_x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.add1([x, attn_output])
        
        ln_x2 = self.layer_norm2(out1)
        # ffn_output.shape: (batch_size, seq_len, d_model)
        # out2.shape      : (batch_size, seq_len, d_model)
        ffn_output = self.ffn(ln_x2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add2([out1, ffn_output])

        return out2





## EncoderModel
class EncoderModel_6(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, sentence_length, rate=0.1, **kwargs):
        super(EncoderModel_6, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.sentence_length = sentence_length
        # self.max_length = max_length

        # self.embedding = keras.layers.Embedding(input_vocab_size, d_model)

        # position_embedding.shape: (1, max_length, d_model)
        # self.position_embedding = get_positional_embedding(max_length, self.d_model)
        self.position_embedding = get_positional_embedding(self.sentence_length, self.d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]  # 列表表达式

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq)
        input_seq_len = tf.shape(x)[1]
        # tf.debugging.assert_less_equal(input_seq_len, max_length,
        #                                "inp_seq_len should be less or equal to self.max_length")

        # x.shape: (batch_size, input_seq_len, d_model)
        # x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # 对 x 做一下缩放
        x += self.position_embedding[:, :input_seq_len, :]  # 只加对应位置的位置编码
        # x += self.position_embedding[:, :511, :]  # 只加对应位置的位置编码


        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        # x.shape: (batch_size, input_seq_len, d_model)
        return x


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            'sentence_length':self.sentence_length
        })
        return config


class DecoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout -> out1
    out1, encoding_outputs -> cross attention -> add & normalize & dropout -> out2
    out2 -> ffn -> add & normalize & dropout -> out3
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layer_norm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        # self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encoding_outputs, training, encoder_decoder_padding_mask): #  decoder_mask,
        # dedcoder_mask: 由look_ahead_mask和decoder_padding_mask合并而来
        # x.shape: (batch_size, target_seq_len, d_model)
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)

        # attn1, out1.shape: (batch_size, target_seq_len, d_model)
        # attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)
        # attn1 = self.dropout1(attn1, training=training)
        # out1 = self.layer_norm1(attn1 + x)

        # attn2, out2.shape: (batch_size, target_seq_len, d_model)
        attn1, attn_weights1 = self.mha1(x, encoding_outputs, encoding_outputs, encoder_decoder_padding_mask)
        attn1 = self.dropout1(attn1, training=training)
        # print(attn1.shape)
        # print(x.shape)
        out1 = self.layer_norm1(attn1 + x)

        # ffn_output, out3.shape: (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(ffn_output + out1)

        return out2, attn_weights1




class DecoderModel_6(keras.layers.Layer):
    def __init__(self, decoder_num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(DecoderModel_6, self).__init__(**kwargs)
        self.d_model = d_model
        self.decoder_num_layers = decoder_num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # self.position_embedding = get_positional_embedding(max_length, d_model)
        self.position_embedding = get_positional_embedding(512, self.d_model)

        self.dropout = keras.layers.Dropout(rate)

        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.decoder_num_layers)]  # 列表表达式

    def call(self, x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        # tf.debugging.assert_less_equal(output_seq_len, self.max_length,
        #                                "out_seq_len should be less or equal to self.max_length")

        attention_weights = {}  # 保存attention_weights保存在字典里

        # x.shape: (batch_size, output_seq_len, d_model)
        # x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.decoder_num_layers):
            x, attn1 = self.decoder_layers[i](x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask)

            attention_weights['decoder_layer{}_attn1'.format(i + 1)] = attn1
            # attention_weights['decoder_layer{}_attn2'.format(i + 1)] = attn2
        # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'decoder_num_layers': self.decoder_num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
        })
        return config
