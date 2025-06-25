"""'Neural Audio Fingerprint for High-specific Audio Retrieval based on
Contrastive Learning', https://arxiv.org/abs/2010.11910"""

import tensorflow as tf

assert tf.__version__ >= "2.0"


class ConvLayer(tf.keras.layers.Layer):
    """Separable convolution layer with 1x3 and 3x1 kernels."""

    def __init__(self, hidden_ch=128, strides=[(1, 1), (1, 1)], norm="layer_norm2d"):
        """
        Arguments
        ---------
            hidden_ch: (int)
            strides: [(int, int), (int, int)]
            norm:
                'layer_norm2d' for normalization on on FxT space (default),
                'layer_norm1d' for normalization on F axis,
                'batch_norm'  for batch-normalization.

        """

        super(ConvLayer, self).__init__()

        self.conv2d_1x3 = tf.keras.layers.Conv2D(
            hidden_ch,
            kernel_size=(1, 3),
            strides=strides[0],
            padding="SAME",
            dilation_rate=(1, 1),
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        self.conv2d_3x1 = tf.keras.layers.Conv2D(
            hidden_ch,
            kernel_size=(3, 1),
            strides=strides[1],
            padding="SAME",
            dilation_rate=(1, 1),
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        if norm == "layer_norm1d":
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)
        elif norm == "layer_norm2d":
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
        elif norm == "batch_norm":
            self.BN_1x3 = tf.keras.layers.BatchNormalization(
                axis=-1
            )  # Fix axis: 2020 Apr20 #TODO::????
            self.BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)
        else:
            raise ValueError(
                "norm must be 'layer_norm1d', 'layer_norm2d', or 'batch_norm'."
            )

        self.layer = tf.keras.Sequential(
            [
                self.conv2d_1x3,
                tf.keras.layers.ELU(),
                self.BN_1x3,
                self.conv2d_3x1,
                tf.keras.layers.ELU(),
                self.BN_3x1,
            ]
        )

    @tf.function
    def call(self, x):
        return self.layer(x)


class DivEncLayer(tf.keras.layers.Layer):
    """
    Multi-head projection a.k.a. 'divide and encode' layer:

    • The concept of 'divide and encode' was discovered in Lai et.al.,
     'Simultaneous Feature Learning and Hash Coding with Deep Neural Networks',
      2015. https://arxiv.org/abs/1504.03410
    • It was also adopted in Gfeller et.al. 'Now Playing: Continuo-
      us low-power music recognition', 2017. https://arxiv.org/abs/1711.10958

    """

    def __init__(self, q=128, unit_dim=[32, 1], norm="layer_norm"):
        """
        Arguments
        ---------
            q: (int) number of slices as 'slice_length = input_dim / q'
            unit_dim: [(int), (int)]
            norm:
                'layer_norm1d' or 'layer_norm2d' uses 1D-layer normalization on the feature.
                'batch_norm' or else uses batch normalization. Default is 'layer_norm2d'.

        """

        super(DivEncLayer, self).__init__()

        self.q = q
        self.unit_dim = unit_dim
        self.norm = norm

        if "layer_norm" in norm:  # Inputs are 2D now
            self.BN = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(q)]
        elif "batch_norm" == norm:
            self.BN = [tf.keras.layers.BatchNormalization(axis=-1) for _ in range(q)]
        else:
            raise ValueError("norm must be a 'layer_norm' or 'batch_norm'.")

        self.split_fc_layers = []
        for i in range(self.q):  # q: num_slices
            self.split_fc_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(self.unit_dim[0], activation="elu"),
                        self.BN[i],
                        tf.keras.layers.Dense(self.unit_dim[1]),
                    ]
                )
            )

    @tf.function
    def _split_encoding(self, x_slices):
        """
        Encode each slice with a separate FC layer and concat the output.

        Input
        -----
            x_slices: (B,Q,S) with Q=num_slices, S=slice length

        Returns
        -------
            encoding: (B,Q)

        """

        out = list()
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        encoding = tf.concat(out, axis=1)  # (B,Q)

        return encoding

    @tf.function
    def call(self, x):
        """
        Split the input into q slices and pass it through the split_encoding layer.

        Input
        -----
            x: (B,1,1,C)

        Returns
        -------
            encoding: (B,Q)

        """

        # x_slices = tf.reshape(x, shape=[x.shape[0], self.q, -1])  # (B,Q*S) -> (B,Q,S)
        batch = tf.shape(x)[0]  # dynamic batch size
        x_slices = tf.reshape(x, [batch, self.q, -1])  # (B,Q*S) -> (B,Q,S)
        encoding = self._split_encoding(x_slices)  # (B,Q)

        return encoding


class FingerPrinter(tf.keras.Model):
    """
    Fingerprinter: 'Neural Audio Fingerprint for High-specific Audio Retrieval
        based on Contrastive Learning', https://arxiv.org/abs/2010.11910

    IN >> [Convlayer]x8 >> [DivEncLayer] >> [L2Normalizer] >> OUT

    """

    def __init__(
        self,
        front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
        front_strides=[
            [(1, 2), (2, 1)],  # F/2,   T/2
            [(1, 2), (2, 1)],  # F/4,   T/4
            [(1, 2), (2, 1)],  # F/8,   T/8
            [(1, 2), (2, 1)],  # F/16,  T/16
            [(1, 1), (2, 1)],  # F/32,  T/16
            [(1, 2), (2, 1)],  # F/64,  T/32
            [(1, 1), (2, 1)],  # F/128, T/32
            [(1, 2), (2, 1)],  # F/256, T/64
        ],
        emb_sz=128,  # q
        fc_unit_dim=[32, 1],
        norm="layer_norm2d",
        mixed_precision=False,
    ):
        """
        Parameters:
        -----------
            front_hidden_ch: (list)
            front_strides: (list)
            emb_sz: (int) default=128
            fc_unit_dim: (list) default=[32,1]
            norm: see ConvLayer for more details.
        """

        super(FingerPrinter, self).__init__()

        assert len(front_hidden_ch) == len(
            front_strides
        ), "front_hidden_ch and front_strides must have the same length."
        assert (
            front_hidden_ch[-1] % emb_sz == 0
        ), "The last hidden channel must be divisible by emb_sz."

        self.front_hidden_ch = front_hidden_ch
        self.front_strides = front_strides
        self.fc_unit_dim = fc_unit_dim
        self.emb_sz = emb_sz
        self.norm = norm
        self.mixed_precision = mixed_precision

        # Front (sep-)conv layers
        self.front_conv = tf.keras.Sequential(name="ConvLayers")
        # Add conv layers
        for i in range(len(front_strides)):
            self.front_conv.add(
                ConvLayer(
                    hidden_ch=front_hidden_ch[i], strides=front_strides[i], norm=norm
                )
            )
        self.front_conv.add(tf.keras.layers.Flatten())  # (B,F',T',C) >> (B,D)

        # Divide & Encoder layer
        self.div_enc = DivEncLayer(q=emb_sz, unit_dim=fc_unit_dim, norm=norm)

    @tf.function
    def call(self, inputs):
        """
        Input
        -----
            x: (B,F,T,1)

        Returns
        -------
            emb: (B,Q)

        """

        # Main processing blocks
        x = self.front_conv(inputs)  # (B,D) with D = (T/2^6) x last_hidden_ch
        x = self.div_enc(x)  # (B,Q)

        # Convert the output to float32 for:
        #   1) avoiding underflow at l2_normalize
        #   2) avoiding underflow at loss calculation
        if self.mixed_precision:
            x = tf.keras.layers.Activation("linear", dtype="float32")(x)

        # L2-normalization of the final embedding
        x = tf.math.l2_normalize(x, axis=1)

        return x
