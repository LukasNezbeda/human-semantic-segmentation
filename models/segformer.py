"""
SegFormer (MiT) B0-like model for binary semantic segmentation.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for stability on some platforms
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore


def _swiglu(x: tf.Tensor) -> tf.Tensor:
	"""Apply the SwiGLU activation to a 2x-expanded tensor."""
	gate, value = tf.split(x, num_or_size_splits=2, axis=-1)
	return tf.nn.silu(gate) * value


class OverlapPatchEmbed(layers.Layer):
	"""Overlap patch embedding with Conv2D + LayerNorm."""

	def __init__(self, embed_dim: int, patch_size: int, stride: int, **kwargs) -> None:
		super().__init__(**kwargs)
		self.proj = layers.Conv2D(
			embed_dim,
			kernel_size=patch_size,
			strides=stride,
			padding="same",
			use_bias=False,
		)
		self.norm = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		x = self.proj(inputs)
		tokens = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
		tokens = self.norm(tokens)
		return tokens, x


class EfficientSelfAttention(layers.Layer):
	"""Multi-head attention with optional spatial reduction."""

	def __init__(self, embed_dim: int, num_heads: int, sr_ratio: int, **kwargs) -> None:
		super().__init__(**kwargs)
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.sr_ratio = sr_ratio
		self.head_dim = embed_dim // num_heads
		self.scale = self.head_dim**-0.5
		self.q = layers.Dense(embed_dim, use_bias=False)
		self.kv = layers.Dense(embed_dim * 2, use_bias=False)
		self.proj = layers.Dense(embed_dim, use_bias=False)
		if sr_ratio > 1:
			self.sr = layers.Conv2D(
				embed_dim,
				kernel_size=sr_ratio,
				strides=sr_ratio,
				padding="same",
				use_bias=False,
			)
			self.norm = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs: tf.Tensor, height: int, width: int) -> tf.Tensor: # type: ignore
		batch_size = tf.shape(inputs)[0]
		token_count = tf.shape(inputs)[1]

		q = self.q(inputs)
		q = tf.reshape(q, [batch_size, token_count, self.num_heads, self.head_dim])
		q = tf.transpose(q, [0, 2, 1, 3])

		if self.sr_ratio > 1:
			x = tf.reshape(inputs, [batch_size, height, width, self.embed_dim])
			x = self.sr(x)
			x = tf.reshape(x, [batch_size, -1, tf.shape(x)[-1]])
			x = self.norm(x)
			kv = self.kv(x)
		else:
			kv = self.kv(inputs)

		kv = tf.reshape(kv, [batch_size, -1, 2, self.num_heads, self.head_dim])
		kv = tf.transpose(kv, [2, 0, 3, 1, 4])
		k, v = kv[0], kv[1]

		attn = tf.matmul(q, k, transpose_b=True) * self.scale
		attn = tf.nn.softmax(attn, axis=-1)

		x = tf.matmul(attn, v)
		x = tf.transpose(x, [0, 2, 1, 3])
		x = tf.reshape(x, [batch_size, token_count, self.embed_dim])
		return self.proj(x)


class MixFFN(layers.Layer):
	"""MLP block with SwiGLU and optional depthwise convolution."""

	def __init__(self, embed_dim: int, mlp_ratio: float, use_dwconv: bool, **kwargs) -> None:
		super().__init__(**kwargs)
		hidden_dim = int(embed_dim * mlp_ratio)
		self.fc1 = layers.Dense(hidden_dim * 2, use_bias=False)
		self.use_dwconv = use_dwconv
		if use_dwconv:
			self.dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
		self.fc2 = layers.Dense(embed_dim, use_bias=False)

	def call(self, inputs: tf.Tensor, height: int, width: int) -> tf.Tensor: # type: ignore
		x = self.fc1(inputs)
		x = _swiglu(x)
		if self.use_dwconv:
			x = tf.reshape(x, [tf.shape(x)[0], height, width, tf.shape(x)[-1]])
			x = self.dwconv(x)
			x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
		return self.fc2(x)


class TransformerBlock(layers.Layer):
	"""SegFormer transformer block with attention and SwiGLU MLP."""

	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		sr_ratio: int,
		mlp_ratio: float,
		use_dwconv: bool,
		**kwargs,
	) -> None:
		super().__init__(**kwargs)
		self.norm1 = layers.LayerNormalization(epsilon=1e-6)
		self.attn = EfficientSelfAttention(embed_dim, num_heads, sr_ratio)
		self.norm2 = layers.LayerNormalization(epsilon=1e-6)
		self.mlp = MixFFN(embed_dim, mlp_ratio, use_dwconv)

	def call(self, inputs: tf.Tensor, height: int, width: int) -> tf.Tensor: # type: ignore
		x = inputs + self.attn(self.norm1(inputs), height=height, width=width) # type: ignore
		x = x + self.mlp(self.norm2(x), height=height, width=width) # type: ignore
		return x


def segformer_b0(shape: tuple[int, int, int]) -> tf.keras.Model:
	"""Build a SegFormer B0-like model for binary segmentation.

	Args:
		shape: Input shape in (H, W, C) format.

	Returns:
		A Keras Model with a single-channel sigmoid output.
	"""
	embed_dims = [32, 64, 160, 256]
	depths = [2, 2, 2, 2]
	num_heads = [1, 2, 5, 8]
	sr_ratios = [8, 4, 2, 1]
	mlp_ratio = 4.0

	inputs = layers.Input(shape)
	x = inputs
	features: list[tf.Tensor] = []

	for idx in range(4):
		patch_size = 7 if idx == 0 else 3
		stride = 4 if idx == 0 else 2
		x, x_map = OverlapPatchEmbed(embed_dims[idx], patch_size, stride)(x)
		height = x_map.shape[1]
		width = x_map.shape[2]

		for _ in range(depths[idx]):
			x = TransformerBlock(
				embed_dims[idx],
				num_heads[idx],
				sr_ratios[idx],
				mlp_ratio,
				use_dwconv=True,
			)(x, height=height, width=width) # type: ignore

		x = layers.Reshape((height, width, embed_dims[idx]))(x)
		features.append(x)

	decoder_dim = 256
	target_height = features[0].shape[1]
	target_width = features[0].shape[2]

	projected = []
	for idx, feat in enumerate(features):
		x = layers.Conv2D(decoder_dim, 1, use_bias=False)(feat)
		if feat.shape[1] != target_height or feat.shape[2] != target_width:
			scale_h = target_height // feat.shape[1] # type: ignore
			scale_w = target_width // feat.shape[2] # type: ignore
			x = layers.UpSampling2D((scale_h, scale_w), interpolation="bilinear")(x)
		projected.append(x)

	x = layers.Concatenate()(projected)
	x = layers.Conv2D(decoder_dim, 1, use_bias=False)(x)
	up_scale_h = shape[0] // target_height # type: ignore
	up_scale_w = shape[1] // target_width # type: ignore
	x = layers.UpSampling2D((up_scale_h, up_scale_w), interpolation="bilinear")(x)
	x = layers.Conv2D(1, 1)(x)
	outputs = layers.Activation("sigmoid")(x)

	return Model(inputs, outputs)


def main() -> None:
	"""Build the model and print its summary."""
	model = segformer_b0((512, 512, 3))
	model.summary()


if __name__ == "__main__":
	main()
