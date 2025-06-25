import tensorflow as tf


class MultiPosConLoss:

    def __init__(self, tau=0.05):
        """NT-Xent a.k.a. Info-NCE loss. It can also works with more than one positive sample per
         anchor i.e., it can implement the SupCon or MultiPosCon losses."""

        self.tau = tau

    @tf.function
    def __call__(self, emb):
        """ emb: hidden vector of shape [Na, Nv, D].
            Na: number of anchors, 
            Nv: number of views (Nv = 1 + Nppa where Nppa is the number of positive samples per anchor)), 
            D: dimension of the hidden vector.
        """

        Na, Nv, D = tf.shape(emb)[0], tf.shape(emb)[1], tf.shape(emb)[2]

        emb = tf.reshape(emb, [Na * Nv, D])  # flatten the tensor
        labels = tf.expand_dims(tf.repeat(tf.range(Na), Nv), -1)

        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        logits_mask = tf.ones_like(mask) - tf.eye(Na * Nv)
        mask = mask * logits_mask  # set the diagonal to zero
        Nppa = tf.reduce_sum(
            mask, axis=1
        )  # positives per anchor is always Nv-1 but we calculate it anyway

        anchor_dot_contrast = tf.divide(tf.matmul(emb, tf.transpose(emb)), self.tau)
        # for numerical stability
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        # compute log_prob
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(
            tf.reduce_sum(exp_logits, axis=1, keepdims=True)
        )

        # Individual loss for each sample
        loss = (-1 / Nppa) * tf.reduce_sum(mask * log_prob, axis=1)  # (Na*Nv,)
        loss = tf.reshape(loss, [Na, Nv])

        # Average over the sample
        loss = tf.reduce_mean(loss)

        return loss
