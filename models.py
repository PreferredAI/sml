import tensorflow as tf


def edist(A, B):
    """
    Compute Euclidean distance between matrix A (NxD) and matrix B (MxD)
    """
    B_transpose = tf.transpose(B)  # (DxM)
    A_norm = tf.reduce_sum(A ** 2, axis=1, keepdims=True)  # (Nx1)
    B_norm = tf.reduce_sum(B_transpose ** 2, axis=0, keepdims=True)  # (1xM)
    A_dot_B = tf.matmul(A, B_transpose)  # (NxM)
    dist = tf.sqrt(A_norm + B_norm - 2 * A_dot_B)  # (NxM)
    return dist


def pairwise_edist(A, B, keepdims=False):
    """
    Compute pairwise Euclidean distance between matrix A (NxD) and matrix B (NxD)
    """
    A_norm = tf.reduce_sum(A ** 2, axis=1, keepdims=keepdims)
    B_norm = tf.reduce_sum(B ** 2, axis=1, keepdims=keepdims)
    pw_A_dot_B = tf.reduce_sum(tf.multiply(A, B), axis=1, keepdims=keepdims)
    dist = tf.sqrt(A_norm + B_norm - 2 * pw_A_dot_B)
    return dist


class SML(tf.keras.Model):
    def __init__(
        self,
        word_emb,
        txt_len,
        img_dim,
        latent_dim,
        act_fn,
        lambda_reg,
    ):
        super(SML, self).__init__()

        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.lambda_reg = lambda_reg
        self.txt_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    word_emb.shape[0],
                    word_emb.shape[1],
                    weights=[word_emb],
                    input_length=txt_len,
                    mask_zero=txt_len != 1,
                    name="WordEmbedding",
                ),
                tf.keras.layers.LSTM(
                    self.latent_dim,
                    kernel_regularizer=tf.keras.regularizers.l2(self.lambda_reg),
                    recurrent_regularizer=tf.keras.regularizers.l2(self.lambda_reg),
                ),
                tf.keras.layers.Dense(
                    self.latent_dim,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(self.lambda_reg),
                ),
            ],
            name="TextEncoder",
        )
        self.img_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(img_dim),
                tf.keras.layers.Dense(
                    self.latent_dim,
                    activation=act_fn,
                    kernel_regularizer=tf.keras.regularizers.l2(self.lambda_reg),
                ),
                tf.keras.layers.Dense(
                    self.latent_dim,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(self.lambda_reg),
                ),
            ],
            name="ImageEncoder",
        )

        self.txt_encoder.summary()
        self.img_encoder.summary()

    def call(self, inputs):
        txt_feat = self.txt_encoder(inputs[0])
        img_feat = self.img_encoder(inputs[1])
        return txt_feat, img_feat


class SMLOppo(SML):
    def __init__(
        self,
        word_emb,
        txt_len,
        img_dim,
        latent_dim,
        act_fn,
        lambda_reg,
        tau,
    ):
        super(SMLOppo, self).__init__(
            word_emb, txt_len, img_dim, latent_dim, act_fn, lambda_reg
        )

        self.tau = tau
        self.s = tf.Variable(tf.random.normal([1, latent_dim], stddev=0.01), name="senti_vec")
        self.s_alpha = tf.keras.layers.Dense(1, activation=tf.nn.softplus)

    @tf.function
    def score(self, txt_input, img_input, senti_input):
        q, p = self.call((txt_input, img_input))
        sign = (
            tf.cast(tf.reshape(senti_input, [-1, 1]), dtype=tf.float32) * 2.0 - 1.0
        )  # convert {0, 1} to {-1, 1}
        alpha = self.s_alpha(q)
        s = sign * alpha * self.s
        scores = -edist(q + s, p)
        return scores

    @tf.function
    def compute_loss(self, x):
        q, p = self.call((x["text"], x["img_feat"]))

        sign = (
            tf.cast(tf.reshape(x["sentiment"], [-1, 1]), dtype=tf.float32) * 2.0 - 1.0
        )  # convert {0, 1} to {-1, 1}
        alpha = self.s_alpha(q)
        s_true = sign * alpha * self.s
        q_true = q + s_true

        q_true_p_dist = edist(q_true, p)
        pw_qp_dist = pairwise_edist(q, p, keepdims=True)
        pw_q_true_p_dist = pairwise_edist(q_true, p, keepdims=True)

        reg = tf.add_n(self.losses)

        loss0 = tf.math.maximum(0.0, pw_q_true_p_dist)
        loss0 = tf.reduce_mean(loss0)

        loss1 = tf.math.maximum(0.0, self.tau + (pw_q_true_p_dist - pw_qp_dist))
        loss1 = tf.reduce_mean(loss1)

        loss2 = tf.math.maximum(0.0, 1.0 + (pw_q_true_p_dist - q_true_p_dist))
        loss2 = tf.reduce_mean(tf.reduce_sum(loss2, axis=1))

        return reg + loss0 + loss1 + loss2


class SMLFlex(SML):
    def __init__(
        self,
        word_emb,
        txt_len,
        img_dim,
        latent_dim,
        act_fn,
        lambda_reg,
        tau,
    ):
        super(SMLFlex, self).__init__(
            word_emb, txt_len, img_dim, latent_dim, act_fn, lambda_reg
        )

        self.tau = tau
        self.senti_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(2, latent_dim, input_length=1, name="SentiVectors"),
                tf.keras.layers.Flatten(),
            ],
            name="SentiEncoder",
        )
        self.senti_encoder.summary()

    @tf.function
    def score(self, txt_input, img_input, senti_input):
        q, p = self.call((txt_input, img_input))
        s = self.senti_encoder(tf.reshape(senti_input, [-1, 1]))
        scores = -edist(q + s, p)
        return scores

    @tf.function
    def compute_loss(self, x):
        q, p = self.call((x["text"], x["img_feat"]))

        s = tf.reshape(x["sentiment"], [-1, 1])
        s_true = self.senti_encoder(s)
        q_true = q + s_true

        q_true_p_dist = edist(q_true, p)
        pw_qp_dist = pairwise_edist(q, p, keepdims=True)
        pw_q_true_p_dist = pairwise_edist(q_true, p, keepdims=True)

        reg = tf.add_n(self.losses)

        loss0 = tf.math.maximum(0.0, pw_q_true_p_dist)
        loss0 = tf.reduce_mean(loss0)

        loss1 = tf.math.maximum(0.0, self.tau + (pw_q_true_p_dist - pw_qp_dist))
        loss1 = tf.reduce_mean(loss1)

        loss2 = tf.math.maximum(0.0, 1.0 + (pw_q_true_p_dist - q_true_p_dist))
        loss2 = tf.reduce_mean(tf.reduce_sum(loss2, axis=1))

        return reg + loss0 + loss1 + loss2
