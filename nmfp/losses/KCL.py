import tensorflow as tf


@tf.function
def gaussian_kernel(x, t=2.0):
    """
    Compute the pairwise potential (energy) based on the Gaussian kernel.

    Args:
        x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
        t (float): Scaling parameter. Default is 2.0.
    """

    # Compute squared pairwise distances using the identity:
    # ||x - y||^2 = ||x||^2 - 2*x.y + ||y||^2
    squared_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # shape: (M, 1)
    squared_dist = (
        squared_norm
        - 2 * tf.matmul(x, x, transpose_b=True)
        + tf.transpose(squared_norm)
    )
    squared_dist = tf.maximum(
        squared_dist, 0.0
    )  # Ensure non-negative due to numerical errors

    # Create a mask to extract only the unique (i < j) distances:
    mask = tf.linalg.band_part(tf.ones_like(squared_dist), 0, -1) - tf.linalg.band_part(
        tf.ones_like(squared_dist), 0, 0
    )
    pairwise_squared = tf.boolean_mask(squared_dist, mask > 0)

    # Compute the Gaussian kernel on these squared distances and average the results.
    kernel_value = tf.reduce_mean(tf.exp(-t * pairwise_squared))

    return kernel_value


@tf.function
def align_gaussian(x, y, t):
    """
    Compute the alignment between anchor points and their positives based on the Gaussian kernel.

    Args:
        x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
        y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
        t (float): Scaling parameter.
    """

    # Compute the Euclidean distance between corresponding pairs (x_i, y_i)
    pairwise_distances = tf.norm(x - y, axis=1)

    return tf.reduce_mean(tf.exp(-t * tf.square(pairwise_distances)))


class KCL:
    """
    Kernel Contrastive Loss (KCL) from https://arxiv.org/abs/2405.18045.

    Args:
        t (float): Kernel hyperparameter. Default is 2.
        kernel (str): Type of kernel to use ('gaussian', 'log', 'imq', 'riesz', 'gaussian_riesz'). Default is 'gaussian'.
        gamma (float): Scaling parameter for the energy loss term. Default is 16.
    """

    def __init__(self, t=2.0, gamma=16.0):

        self.t = t
        self.gamma = gamma

    @tf.function
    def __call__(self, emb):
        """
        Forward pass for the KCL loss calculation.

        Args:
            z (Tensor): Input tensor of shape (2M, d) where M is the batch size and d is the embedding dimension.
        """

        emb_org, emb_rep = emb[:, 0], emb[:, 1]

        energy = gaussian_kernel(emb_org, self.t) + gaussian_kernel(emb_rep, self.t)
        alignment = 2 * align_gaussian(emb_org, emb_rep, self.t)

        loss = -alignment + tf.cast(self.gamma, energy.dtype) * energy
        return loss
