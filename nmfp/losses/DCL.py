import tensorflow as tf


class DCL:
    """
    Decoupled Contrastive Loss (DCL) implementation in TensorFlow.

    This version computes the loss in both directions: (real, augmented) and (augmented, real).
    It then averages the two for a symmetric loss.

    Args:
        tau: tau scaling factor.
    """

    def __init__(self, tau=0.05):

        self.tau = tau

    @tf.function
    def _one_way_loss(self, a, b):
        """
        Compute one-way loss using view a as reference and b as the second view.
        """

        cross_similarity = tf.matmul(a, b, transpose_b=True)
        same_similarity = tf.matmul(a, a, transpose_b=True)

        pos_pairs_loss = -1 * tf.linalg.diag_part(cross_similarity) / self.tau

        all_similarity = (
            tf.concat([same_similarity, cross_similarity], axis=1) / self.tau
        )

        # Create a mask to exclude the positive pairs' similarities from the denominator
        # i.e. decoupling
        pos_mask = tf.tile(tf.eye(tf.shape(a)[0], dtype=a.dtype), [1, 2])
        # _inf = tf.constant(-float("inf"), dtype=a.dtype)
        mask_value = tf.constant(-1e9, dtype=a.dtype)
        pos_mask = pos_mask * mask_value

        # Compute the negative loss using the log-sum-exp trick.
        neg_pairs_loss = tf.reduce_logsumexp(all_similarity + pos_mask, axis=1)

        return tf.reduce_mean(pos_pairs_loss + neg_pairs_loss)

    @tf.function
    def __call__(self, emb):
        """
        Compute the symmetric decoupled contrastive loss by averaging the one-way losses.

        Args:
            z1: Tensor of shape (batch_size, embedding_dim), first view embeddings.
            z2: Tensor of shape (batch_size, embedding_dim), second view embeddings.

        Returns:
            A scalar tensor representing the averaged loss.
        """

        emb_org, emb_rep = emb[:, 0], emb[:, 1]

        loss_1 = self._one_way_loss(emb_org, emb_rep)
        loss_2 = self._one_way_loss(emb_rep, emb_org)
        return loss_1 + loss_2
