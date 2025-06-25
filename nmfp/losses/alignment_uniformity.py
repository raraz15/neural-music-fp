import tensorflow as tf


class AlignmentLoss:

    def __init__(
        self,
        alpha: float = 2,
    ):
        """Initialize the AlignmentLoss.

        Parameters:
        -----------
        alpha: float
            The exponent used to scale the norm of the difference in alignment loss.
        """

        assert alpha > 0, "Expected a positive exponent for the alignment loss."
        self.alpha = alpha

    @tf.function
    def __call__(self, emb_org, emb_rep):

        # Calculate Euclidean distance
        distance = tf.norm(emb_org - emb_rep, ord="euclidean", axis=1)
        # Return the mean of the distances raised to the power of alpha
        return tf.reduce_mean(tf.pow(distance, self.alpha))


class UniformityLoss:
    def __init__(self, t: float = 2):
        """Initialize the UniformityLoss.

        Parameters:
        -----------
        t: float
            The temperature used in the uniformity loss.
        """

        assert t > 0, "Expected a positive temperature for the uniformity loss."
        self.t = t

    @tf.function
    def __call__(self, emb):

        # Compute pairwise distances using broadcasting
        diff = tf.expand_dims(emb, 1) - tf.expand_dims(emb, 0)
        pairwise_dist = tf.norm(diff, axis=-1)

        # Create a mask to exclude the diagonal (self distances)
        mask = 1 - tf.eye(tf.shape(pairwise_dist)[0])
        # Apply the mask: this sets diagonal elements to 0 and only averages over non-diagonal entries
        # We also need to adjust for the number of elements we are averaging over.
        num_elements = tf.reduce_sum(mask)
        exp_component = tf.exp(-self.t * tf.pow(pairwise_dist, 2)) * mask
        mean_exp = tf.reduce_sum(exp_component) / num_elements

        return tf.math.log(mean_exp)


class AlignmentUniformityLoss:

    def __init__(
        self,
        alpha: float = 2,
        t: float = 2,
        w_alignment: float = 1,
        w_uniformity: float = 1,
    ):
        """Initialize the AlignmentUniformityLoss.

        Parameters:
        -----------
        alpha: float
            The exponent used to scale the norm of the difference in alignment loss.
        t: float
            The temperature used in the uniformity loss.
        w_alignment: float
            The weight of the alignment loss.
        w_uniformity: float
            The weight of the uniformity loss.
        """

        assert w_alignment > 0, "Expected a positive weight for alignment loss."
        assert w_uniformity > 0, "Expected a positive weight for uniformity loss."

        self.alignment_loss = AlignmentLoss(alpha)
        self.uniformity_loss = UniformityLoss(t)
        self.w_alignment = w_alignment
        self.w_uniformity = w_uniformity

    @tf.function
    def __call__(self, emb):

        emb_org, emb_rep = emb[:, 0], emb[:, 1]

        alignment_loss = self.alignment_loss(emb_org, emb_rep)
        uniformity_loss = (
            self.uniformity_loss(emb_org) + self.uniformity_loss(emb_rep)
        ) / 2

        return self.w_alignment * alignment_loss + self.w_uniformity * uniformity_loss
