import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore



""" Building blocks """
class ConvMlp(layers.Layer):
    """Conv-MLP: 1x1 -> DW 3x3 -> GELU -> Dropout -> 1x1 -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = layers.Conv2D(hidden_channels, 1, use_bias=False)
        self.dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.act = layers.Activation("gelu")
        self.drop1 = layers.Dropout(drop)
        self.fc2 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.drop2 = layers.Dropout(drop)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop1(x, training=training) #type: ignore
        x = self.fc2(x)
        x = self.drop2(x, training=training) #type: ignore
        return x

class StemConv(layers.Layer):
    """Two 3x3 convs with stride 2 each (downsample /4)."""

    def __init__(self, out_channels: int, norm: str = "ln", **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channels // 2, 3, strides=2, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(out_channels, 3, strides=2, padding="same", use_bias=False)

        if norm == "bn":
            self.norm1 = layers.BatchNormalization(axis=-1)
            self.norm2 = layers.BatchNormalization(axis=-1)
        else:
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.act = layers.Activation("gelu")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.conv1(x)
        x = self.norm1(x, training=training) if isinstance(self.norm1, layers.BatchNormalization) else self.norm1(x) #type: ignore
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training) if isinstance(self.norm2, layers.BatchNormalization) else self.norm2(x) #type: ignore
        return x

class MSCA(layers.Layer):
    """Multi-Scale Convolutional Attention (depthwise large kernels)."""

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # depthwise conv branches; 'same' padding matches odd-kernel padding behavior.
        self.conv0 = layers.DepthwiseConv2D(5, padding="same", use_bias=False)

        self.conv0_1 = layers.DepthwiseConv2D((1, 7), padding="same", use_bias=False)
        self.conv0_2 = layers.DepthwiseConv2D((7, 1), padding="same", use_bias=False)

        self.conv1_1 = layers.DepthwiseConv2D((1, 11), padding="same", use_bias=False)
        self.conv1_2 = layers.DepthwiseConv2D((11, 1), padding="same", use_bias=False)

        self.conv2_1 = layers.DepthwiseConv2D((1, 21), padding="same", use_bias=False)
        self.conv2_2 = layers.DepthwiseConv2D((21, 1), padding="same", use_bias=False)

        self.conv3 = layers.Conv2D(channels, 1, use_bias=False)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        u = x
        attn = self.conv0(x)

        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u

class SpatialAttention(layers.Layer):
    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proj_1 = layers.Conv2D(channels, 1, use_bias=False)
        self.act = layers.Activation("gelu")
        self.msca = MSCA(channels)
        self.proj_2 = layers.Conv2D(channels, 1, use_bias=False)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj_2(x)
        return x + shortcut

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if (not training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [tf.shape(x)[0], 1, 1, 1]
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(random_tensor)
        return (x / keep_prob) * binary_mask

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg

class SegNeXtBlock(layers.Layer):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-2,
        norm: str = "ln",  # "ln" recommended for small batch
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if norm == "bn":
            self.norm1 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
            self.norm2 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
        else:
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.attn = SpatialAttention(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = ConvMlp(channels, hidden_channels=hidden, drop=drop)

        self.drop_path = StochasticDepth(drop_path)

        # Per-channel layer scale (broadcast across H,W)
        self.gamma1 = self.add_weight(
            name="gamma1",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(layer_scale_init),
            trainable=True,
        )
        self.gamma2 = self.add_weight(
            name="gamma2",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(layer_scale_init),
            trainable=True,
        )

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        a = self.attn(self.norm1(x), training=training) #type: ignore
        a = a * self.gamma1[None, None, None, :]
        x = x + self.drop_path(a, training=training) #type: ignore

        m = self.mlp(self.norm2(x), training=training) #type: ignore
        m = m * self.gamma2[None, None, None, :]
        x = x + self.drop_path(m, training=training) #type: ignore
        return x

class OverlapDownsample(layers.Layer):
    """Conv downsample between stages (rough OverlapPatchEmbed analogue)."""

    def __init__(self, out_channels: int, norm: str = "ln", **kwargs) -> None:
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(out_channels, 3, strides=2, padding="same", use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6) if norm == "ln" else layers.BatchNormalization(axis=-1)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.proj(x)
        return self.norm(x, training=training) if isinstance(self.norm, layers.BatchNormalization) else self.norm(x) #type: ignore



""" Model Building """

def segnext(
    input_shape: tuple[int, int, int],
    embed_dims: list[int] = [32, 64, 160, 256],
    depths: list[int] = [3, 3, 5, 2],
    mlp_ratios: list[float] = [8.0, 8.0, 4.0, 4.0],
    decoder_dim: int = 128,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    norm: str = "ln",
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    """Build the SegNeXt model.

    Args:
        input_shape: The shape of the input tensor (height, width, channels).

    Returns:
        A Keras Model instance representing the SegNeXt architecture.
    """
    inputs = layers.Input(shape=input_shape)

    # 1) Stem: (H,W) -> (H/4,W/4)
    x = StemConv(embed_dims[0], norm=norm, name="stem")(inputs)

    # Stochastic depth schedule (linearly increasing across blocks)
    total_blocks = sum(depths)
    dpr = tf.linspace(0.0, drop_path_rate, total_blocks)
    dpr_idx = 0

    features: list[tf.Tensor] = []

    # 2) Stages
    for stage_idx in range(4):
        if stage_idx > 0:
            x = OverlapDownsample(embed_dims[stage_idx], norm=norm, name=f"down{stage_idx+1}")(x)

        for block_idx in range(depths[stage_idx]):
            x = SegNeXtBlock(
                channels=embed_dims[stage_idx],
                mlp_ratio=mlp_ratios[stage_idx],
                drop=drop_rate,
                drop_path=float(dpr[dpr_idx].numpy()),  # OK since input_shape is static in your scripts
                norm=norm,
                name=f"s{stage_idx+1}_b{block_idx+1}",
            )(x)
            dpr_idx += 1

        features.append(x)  # stage outputs at /4, /8, /16, /32

    # 3) Decoder (simple FPN-style)
    # target = highest resolution feature (stage 1, /4)
    target_h = features[0].shape[1]
    target_w = features[0].shape[2]
    if target_h is None or target_w is None:
        raise ValueError("Static spatial dims required (use fixed input_shape like (512,1024,3)).")

    projected: list[tf.Tensor] = []
    for i, feat in enumerate(features):
        p = layers.Conv2D(decoder_dim, 1, use_bias=False, name=f"dec_proj{i+1}")(feat)
        if feat.shape[1] != target_h or feat.shape[2] != target_w:
            p = layers.Resizing(target_h, target_w, interpolation="bilinear", name=f"dec_up{i+1}")(p)
        projected.append(p)

    x = layers.Concatenate(name="dec_concat")(projected)
    x = layers.Conv2D(decoder_dim, 1, use_bias=False, name="dec_fuse")(x)

    # Upsample back to input resolution and predict mask
    x = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear", name="out_up")(x)
    x = layers.Conv2D(1, 1, name="out_conv")(x)
    outputs = layers.Activation("sigmoid", name="out_sigmoid")(x)

    return Model(inputs, outputs, name="segnext_tiny")



def main() -> None:
    """Build the model and print its summary."""
    model = segnext((512, 512, 3))
    model.summary()

if __name__ == "__main__":
    main()