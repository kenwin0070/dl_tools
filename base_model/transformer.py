# coding=utf-8

import gc
import warnings
import tensorflow as tf
import datatable as dtable
import tensorflow.keras.backend as K
warnings.filterwarnings('ignore')
tf.random.set_seed(42)


def transformer_encoder():
    import tensorflow_addons as tfa

    device = 'GPU' if 'GPU' in tf.test.gpu_device_name() else 'CPU/TPU'
    print('Device:', device)

    if device == 'GPU':
        pass

    print("Tensorflow version " + tf.__version__)
    AUTO = tf.data.experimental.AUTOTUNE

    def scaled_dot_product_attention(q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    class MultiHeadAttention(tf.keras.layers.Layer):

        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)  # (batch_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, seq_len, d_model)

            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

            # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
            # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

            concat_attention = tf.reshape(scaled_attention,
                                          (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

            output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

            return output, attention_weights

    def point_wise_feed_forward_network(d_model, dff):

        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='swish'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    class EncoderLayer(tf.keras.layers.Layer):

        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(EncoderLayer, self).__init__()

            self.mha = MultiHeadAttention(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)

            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

            ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

            return out2

    class TransformerEncoder(tf.keras.layers.Layer):

        def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
            """

            :param num_layers:                  transformer_encoder 层数
            :param d_model:                     transformer_encoder 输入数据维度
            :param num_heads:                   头数（多头注意力机制）
            :param dff:                         feed_forward 中间层维度大小
            :param maximum_position_encoding:   输入数据窗口大小
            :param rate:                        神经元失活概率
            """
            super(TransformerEncoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.dff = dff
            self.maximum_position_encoding = maximum_position_encoding
            self.rate = rate

            self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maximum_position_encoding,
                                                     output_dim=self.d_model)

            self.encoder_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                                   for _ in range(self.num_layers)]

            self.dropout = tf.keras.layers.Dropout(self.rate)

        def call(self, x, training, mask=None):
            seq_len = tf.shape(x)[1]

            # adding embedding and position encoding.
            positions = tf.range(start=0, limit=seq_len, delta=1)
            x += self.pos_emb(positions)

            x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x = self.encoder_layers[i](x, training, mask)

            return x  # (batch_size, input_seq_len, d_model)

    def create_transformer_model(input_columns, input_window_size, output_labels, num_layers, d_model, num_heads, dff,
                                 dropout_rate, weight_decay, learning_rate, label_smoothing):
        """
        创建 transformer encoder 模型
        :param input_columns:       输入数据维度
        :param input_window_size:   输入数据窗口大小
        :param output_labels:       输出目标 单目标/多目标
        :param num_layers:          transformer_encoder 层数
        :param d_model:             transformer_encoder 输入数据维度
        :param num_heads:           头数（多头注意力机制）
        :param dff:                 feed_forward 中间层维度大小
        :param dropout_rate:        神经元失活概率
        :param weight_decay:        学习算法：学习率衰减
        :param learning_rate:       学习算法：学习率
        :param label_smoothing:
        :return:
        """
        inp = tf.keras.layers.Input(shape=(input_window_size, input_columns))
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dense(d_model)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = TransformerEncoder(num_layers, d_model, num_heads, dff, input_window_size, dropout_rate)(x)
        out = tf.keras.layers.Dense(output_labels, activation='sigmoid')(x[:, -1, :])

        transformer_model = tf.keras.models.Model(inputs=inp, outputs=out)
        transformer_model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=learning_rate),
                                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                                  metrics=tf.keras.metrics.AUC(name='AUC'),
                                  )

        return transformer_model

    # Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is set.
        # On Kaggle this is always the case.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    batch_size = 4096 * strategy.num_replicas_in_sync
    train_num_layers = 1
    train_d_model = 96
    train_num_heads = 1
    train_dff = 64
    window_size = 3
    train_dropout_rate = 0.15
    train_weight_decay = 0
    train_label_smoothing = 1e-2
    train_learning_rate = 1e-3 * strategy.num_replicas_in_sync
    verbose = 1

    print('Loading...')
    train = dtable.fread('../input/jane-street-market-prediction/train.csv').to_pandas()
    features = [c for c in train.columns if 'feature' in c]
    print('Filling...')
    train = train.query('weight > 0').reset_index(drop=True)
    train[features] = train[features].fillna(method='ffill').fillna(0)
    train['action'] = (train['resp'] > 0).astype('int')
    print('Finish.')

    with strategy.scope():
        model = create_transformer_model(len(features), window_size, 1, train_num_layers, train_d_model, train_num_heads, train_dff,
                                         train_dropout_rate, train_weight_decay, train_learning_rate, train_label_smoothing)
    model.summary()

    K.clear_session()
    del model
    rubbish = gc.collect()


if __name__ == '__main__':
    print(1)

