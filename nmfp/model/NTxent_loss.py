""" Simple implementation of Normalized Temperature Crossentropy loss for 
single GPU. 

Taken from https://github.com/mimbres/neural-audio-fp

Input batch for FP training:
    • We assume a batch of ordered embeddings as {a0, a1,...b0, b1,...}.
    • In SimCLR paper, a(i) and b(i) are augmented samples from the ith
      original sample.
    • In our Fingerprinter, we assume a(i) is ith original sample, while b(i) 
      is augmented samples from a(i).
    • In any case, input embeddings should be split by part a and b.

How is it different from SimCLR author's code?
    • The drop_diag() part gives the better readability, and is coceptually
      more making sense (in my opinion).
    • Other than that, it is basically equivalent.

"""

import tensorflow as tf


class NTxentLoss:
    def __init__(self, n_org: int, n_rep: int, tau=0.05, **kwargs):
        """Init."""

        self.n_org = n_org
        self.n_rep = n_rep
        self.tau = tau

        """ Generate temporal labels and diag masks. """
        self.labels = tf.one_hot(tf.range(n_org), n_org * 2 - 1)
        self.mask_not_diag = tf.constant(tf.cast(1 - tf.eye(n_org), tf.bool))

    @tf.function
    def drop_diag(self, x):
        x = tf.boolean_mask(x, self.mask_not_diag)
        return tf.reshape(x, (self.n_org, self.n_org - 1))

    @tf.function
    def compute_loss(self, emb_org, emb_rep):
        """NTxent Loss function for neural audio fingerprint.

        • Every input embeddings must be L2-normalized...
        • Batch-size must be an even number.

        Args
        ----
        emb_org: tensor of shape (nO, d)
            nO is the number of original samples. d is dimension of embeddings.
        emb_rep: tensor of shape (nR, d)
            nR is the number of replica (=augmented) samples.

        Returns
        -------
            (loss, sim_mtx, labels)

        """
        ha, hb = emb_org, emb_rep  # assert(len(emb_org)==len(emb_rep))
        logits_aa = tf.matmul(ha, ha, transpose_b=True) / self.tau
        logits_aa = self.drop_diag(logits_aa)  # modified
        logits_bb = tf.matmul(hb, hb, transpose_b=True) / self.tau
        logits_bb = self.drop_diag(logits_bb)  # modified
        logits_ab = tf.matmul(ha, hb, transpose_b=True) / self.tau
        logits_ba = tf.matmul(hb, ha, transpose_b=True) / self.tau
        loss_a = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ab, logits_aa], 1)
        )
        loss_b = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ba, logits_bb], 1)
        )
        return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), self.labels
