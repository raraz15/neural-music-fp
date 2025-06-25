import tensorflow as tf


@tf.function
def pairwise_distance(x, y=None, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || x[i, :] - y[j, :] ||_2

    Args:
      x: 2-D Tensor of size [N, D].
      y (optional): 2-D Tensor of size [M, D].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [N, M].
    """

    if y is None:
        y = x
        is_equal = True
    else:
        # In case the user provides the same matrix
        is_equal = x is y

    y = tf.transpose(y)
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(x), axis=[1], keepdims=True),
        tf.math.reduce_sum(tf.math.square(y), axis=[0], keepdims=True),
    ) - 2.0 * tf.matmul(x, y)

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        # Deal with zeroes before the sqrt
        error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )
        pairwise_distances = tf.math.multiply(
            pairwise_distances,
            tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
        )

    # Explicitly set diagonals to zero
    if is_equal:
        mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
            tf.ones([tf.shape(x)[0]])
        )
        pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


@tf.function
def hard_negative_mining(distance_matrix: tf.Tensor, mask: tf.Tensor):
    """
    For each row in the distance matrix, select the index of the hardest negative sample,
    i.e. the one with the smallest distance among those where mask==1.

    Parameters:
    -----------
    distance_matrix: tf.Tensor
        2D tensor of shape (m, n) where distance_matrix[i, j] is the distance between
        the i-th sample and the j-th sample.
    mask: tf.Tensor
        2D tensor of shape (m, n) (typically float32 or bool) indicating valid negatives.
        Valid entries should be 1 (or True) and invalid entries 0 (or False).

    Returns:
    --------
    anchor_neg_distances: tf.Tensor
        1D tensor of shape (m,) with the selected (minimum) distance for each anchor.
    """

    # We assume that invalid candidates (mask==0) should be ignored.
    # To do so, we add a large constant to distances where mask==0.
    INF = tf.constant(1e9, dtype=distance_matrix.dtype)
    mask = tf.cast(mask, distance_matrix.dtype)
    masked_distances = distance_matrix + (1.0 - mask) * INF

    # For each row, the hardest negative is the one with the minimum masked distance.
    anchor_neg_distances = tf.reduce_min(masked_distances, axis=1)

    return anchor_neg_distances


@tf.function
def hard_positive_mining(distance_matrix: tf.Tensor, mask: tf.Tensor):
    """
    For each row in the distance matrix, select the index of the hardest positive sample,
    i.e. the one with the largest distance among those where mask==1.

    Parameters:
    -----------
    distance_matrix: tf.Tensor
        2D tensor of shape (m, n) where distance_matrix[i, j] is the distance between
        the i-th anchor and the j-th sample.
    mask: tf.Tensor
        2D tensor of shape (m, n) indicating valid positives.
        Valid entries should be 1 (or True) and invalid entries 0 (or False).

    Returns:
    --------
    anchor_pos_distances: tf.Tensor
        1D tensor of shape (m,) with the selected (maximum) distance for each anchor.
    """

    # We assume that invalid candidates (mask==0) should be ignored.
    # To do so, we add a large constant to distances where mask==0.
    INF = tf.constant(1e9, dtype=distance_matrix.dtype)
    mask = tf.cast(mask, distance_matrix.dtype)
    masked_distances = distance_matrix - (1.0 - mask) * INF

    # For each row, the hardest positive is the one with the maximum masked distance.
    anchor_pos_distances = tf.reduce_max(masked_distances, axis=1)

    return anchor_pos_distances


@tf.function
def random_positive_mining(distance_matrix: tf.Tensor, mask_pos: tf.Tensor):
    """
    Randomly sample a positive sample for each anchor from the given distance matrix
    and positive mask.

    Parameters:
    -----------
    distance_matrix : tf.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        the i-th anchor and the j-th candidate.
    mask_pos : tf.Tensor
        2D tensor of shape (n, n) indicating valid positive samples.
        Valid entries should be 1 (or True) and invalid ones 0 (or False).

    Returns:
    --------
    anchor_pos_distances : tf.Tensor
        1D tensor of shape (n,), distances between the anchors and their selected positives.
    """

    # Convert mask to boolean if not already
    mask_bool = tf.cast(mask_pos, tf.bool)

    # Build logits: assign 0 for valid positions and a large negative number for invalid ones.
    logits = tf.where(
        mask_bool,
        tf.zeros_like(mask_pos, dtype=tf.float32),
        tf.fill(tf.shape(mask_pos), -1e9),
    )

    # Sample one index per row using the logits (this acts like torch.multinomial for binary masks)
    positive_indices = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
    positive_indices = tf.squeeze(positive_indices, axis=1)  # Shape: (n,)

    # Gather the corresponding distances for each anchor and its chosen positive sample.
    anchor_pos_distances = tf.gather(distance_matrix, positive_indices, batch_dims=1)

    return anchor_pos_distances


@tf.function
def easy_positive_mining(distance_matrix, mask_pos):
    """
    For each row in the distance matrix, select the index of the easiest positive sample,
    i.e. the one with the smallest distance among those where mask==1.

    Parameters:
    -----------
    distance_matrix: tf.Tensor
        2D tensor of shape (m, n) where distance_matrix[i, j] is the distance between
        the i-th anchor and the j-th sample.
    mask: tf.Tensor
        2D tensor of shape (m, n) indicating valid positives.
        Valid entries should be 1 (or True) and invalid entries 0 (or False).

    Returns:
    --------
    anchor_pos_distances: tf.Tensor
        1D tensor of shape (m,) with the selected (minimum) distance for each anchor.
    """

    # We assume that invalid candidates (mask==0) should be ignored.
    # To do so, we add a large constant to distances where mask==0.
    INF = tf.constant(1e9, dtype=distance_matrix.dtype)
    mask_pos = tf.cast(mask_pos, distance_matrix.dtype)
    masked_distances = distance_matrix + (1.0 - mask_pos) * INF

    # For each row, the hardest negative is the one with the minimum masked distance.
    anchor_pos_distances = tf.reduce_min(masked_distances, axis=1)

    return anchor_pos_distances


@tf.function
def semi_hard_negative_mining(
    distance_matrix,
    dist_AP,
    mask_neg,
    margin,
):
    """
    Given a pairwise distance matrix of all the anchors and a mask that indicates the
    possible indices for sampling negatives, mine the semi-hard negative sample for each
    anchor. All samples are treated as anchor points. If there are no possible semi-hard
    negatives, sample randomly.

    Parameters:
    -----------
    distance_matrix: tf.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j].
    dist_AP: tf.Tensor
        1D tensor of shape (n,) where dist_AP[i] is the distance between the i-th anchor
        and its positive sample.
    mask_neg: tf.Tensor
        A tensor (or boolean-convertible) mask of shape (n, n) indicating valid negatives.
    margin: float
        Margin for the triplet loss.

    Returns:
    --------
    anchor_neg_distances: tf.Tensor
        1D tensor of shape (n,) containing the distances between the anchors and their
        chosen negative samples.
    """

    # Create mask for semi-hard negatives:
    # Condition: dist_AP < distance_matrix < (dist_AP + margin)
    dist_AP_exp = tf.expand_dims(dist_AP, axis=1)  # shape: (n, 1)
    mask_semi_hard_neg = tf.cast(
        tf.logical_and(
            tf.logical_and(
                dist_AP_exp < distance_matrix, distance_matrix < (dist_AP_exp + margin)
            ),
            tf.cast(mask_neg, tf.bool),
        ),
        tf.float32,
    )

    # Determine which anchors have no semi-hard negatives
    sum_semi_hard = tf.reduce_sum(mask_semi_hard_neg, axis=1)  # shape: (n,)
    empty_hollow_sphere = tf.equal(sum_semi_hard, 0.0)
    non_empty_hollow_sphere = tf.logical_not(empty_hollow_sphere)

    # Initialize output tensors (anchor-negative distances and negative indices)
    dist_AN = tf.zeros_like(dist_AP)

    # Process anchors with at least one semi-hard negative:
    if tf.reduce_any(non_empty_hollow_sphere):
        dm_non_empty = tf.boolean_mask(distance_matrix, non_empty_hollow_sphere)
        mask_semi_non_empty = tf.boolean_mask(
            mask_semi_hard_neg, non_empty_hollow_sphere
        )

        anchor_neg_distances_with = hard_negative_mining(
            dm_non_empty, mask_semi_non_empty
        )
        # Get indices of anchors with non-empty hollow sphere
        non_empty_idx = tf.where(non_empty_hollow_sphere)[:, 0]
        dist_AN = tf.tensor_scatter_nd_update(
            dist_AN, tf.expand_dims(non_empty_idx, 1), anchor_neg_distances_with
        )

    # Process anchors with no semi-hard negatives:
    if tf.reduce_any(empty_hollow_sphere):
        dm_empty = tf.boolean_mask(distance_matrix, empty_hollow_sphere)
        # Here we use the full negative mask (mask_neg) for these anchors.
        mask_neg_empty = tf.boolean_mask(
            tf.cast(mask_neg, tf.float32), empty_hollow_sphere
        )

        anchor_neg_distances_without = hard_negative_mining(dm_empty, mask_neg_empty)
        empty_idx = tf.where(empty_hollow_sphere)[:, 0]
        dist_AN = tf.tensor_scatter_nd_update(
            dist_AN, tf.expand_dims(empty_idx, 1), anchor_neg_distances_without
        )

    return dist_AN


class TripletLoss:
    def __init__(self, margin, pos_mode="hard", neg_mode="semi-hard", squared=False):
        """
        Args:
            margin: margin for triplet loss.
            squared: Boolean. If True, uses squared Euclidean distances.
        """

        self.margin = margin
        self.squared = squared
        self.pos_mode = pos_mode
        self.neg_mode = neg_mode

    @tf.function
    def __call__(self, emb):
        """
        Expects emb of shape (Na, Nv, d) where:
          • emb[:,0] are anchors,
          • emb[:,1:] are the corresponding positives.

          Nv = 1 + N_ppa
        """

        N_a, N_v = tf.shape(emb)[0], tf.shape(emb)[1]

        # Create the SSL masks. Anchor major reshaping for faster computation
        y_labels = tf.tile(tf.range(N_a), [N_v])  # shape: (Na * (1 +N_ppa),)
        mask_pos = self._get_anchor_positive_triplet_mask(y_labels)
        mask_neg = self._get_anchor_negative_triplet_mask(y_labels)

        distances = self.compute_distances(emb)

        if self.pos_mode == "hard":
            D_AP = hard_positive_mining(distances, mask_pos)
        elif self.pos_mode == "random":
            D_AP = random_positive_mining(distances, mask_pos)
        elif self.pos_mode == "easy":
            D_AP = easy_positive_mining(distances, mask_pos)
        else:
            raise ValueError(f"Invalid positive mining mode: {self.pos_mode}")

        if self.neg_mode == "hard":
            D_AN = hard_negative_mining(distances, mask_neg)
        elif self.neg_mode == "semi-hard":
            D_AN = semi_hard_negative_mining(
                distances,
                D_AP,
                mask_neg,
                self.margin,
            )
        else:
            raise ValueError(f"Invalid negative mining mode: {self.neg_mode}")

        # Compute loss
        loss = tf.maximum(D_AP - D_AN + self.margin, 0.0)

        # Average over the batch
        loss = tf.reduce_mean(loss)

        return loss

    @tf.function
    def compute_distances(self, emb):

        N_a, Nv = tf.shape(emb)[0], tf.shape(emb)[1]
        # Anchor major reshaping for faster computation
        emb_flat = tf.transpose(emb, perm=[1, 0, 2])  # shape: (Nv, Na, d)
        emb_flat = tf.reshape(emb_flat, (N_a * Nv, -1))
        ta = tf.TensorArray(tf.float32, size=Nv)
        # Iterate using tf.range so that the loop remains in graph mode.
        for i in tf.range(Nv):
            dist = pairwise_distance(
                emb[:, i], emb_flat, squared=self.squared
            )  # (Na, Na*Nv)
            ta = ta.write(i, dist)
        distances = ta.concat()  # shape: (Na*Nv, Na*Nv)
        return distances

    @tf.function
    def _get_anchor_positive_triplet_mask(self, labels):
        # Create a mask where mask[a, p] is True if a and p are distinct and have the same label.
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        return tf.logical_and(indices_not_equal, labels_equal)

    @tf.function
    def _get_anchor_negative_triplet_mask(self, labels):
        # Create a mask where mask[a, n] is True if a and n have different labels.
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        return tf.logical_not(labels_equal)
