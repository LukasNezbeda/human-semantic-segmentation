import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore

def conv_bn_relu(x: tf.Tensor, filters: int) -> tf.Tensor:
    """A helper function to apply Conv2D, BatchNormalization, and ReLU activation.

    Args:
        x: Input tensor.
        filters: Number of filters for the Conv2D layers.

    Returns:
        Output tensor after applying the convolutional block.
    """

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """A convolutional block consisting of two Conv2D layers with ReLU activation.

    Args:
        x: Input tensor.
        filters: Number of filters for the Conv2D layers.

    Returns:
        Output tensor after applying the convolutional block.
    """

    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    x = layers.Activation("relu")(x)
    return x



def pooling(x: tf.Tensor, factor: int) -> tf.Tensor:
    """A helper function to apply MaxPooling2D with a specified factor.

    Args:
        x: Input tensor.
        factor: Pooling factor (e.g., 2 for 1/2, 4 for 1/4).

    Returns:
        Output tensor after applying the pooling operation.
    """
    if factor == 1:
        return x
    
    return layers.MaxPool2D(pool_size=(factor, factor), strides=(factor, factor))(x)

def upsample(x: tf.Tensor, factor: int) -> tf.Tensor:
    """A helper function to apply UpSampling2D with a specified factor.

    Args:
        x: Input tensor.
        factor: Upsampling factor (e.g., 2 for 2x, 4 for 4x).

    Returns:
        Output tensor after applying the upsampling operation.
    """
    if factor == 1:
        return x
    
    return layers.UpSampling2D(size=(factor, factor), interpolation="bilinear")(x)



def decoder_stage(inputs: list[tf.Tensor], filters: int) -> tf.Tensor:
    """A decoder stage for full-scale aggregation.

    Args:
        inputs: List of input tensors from different levels.
        filters: Number of filters for the Conv2D layers.

    Returns:
        Output tensor after applying the decoder stage.
    """
    
    projected = [conv_bn_relu(t, filters) for t in inputs]
    x = layers.Concatenate()(projected)
    x = conv_bn_relu(x, filters)
    return x



def unet3_plus(
    shape: tuple[int, int, int],
    base_filters: int = 24,
    decoder_filters: int = 48,
) -> tf.keras.Model:
    """Build a UNet3+ model for binary segmentation.

    Args:
        shape: Input shape in (H, W, C) format.
        base_filters: Number of filters in the first encoder layer.
        decoder_filters: Number of filters in the decoder layers.

    Returns:
        A UNet3+ model for binary segmentation.
    """
    inputs = layers.Input(shape)

    # Encoder (5 levels)
    e1 = conv_block(inputs, base_filters)                           #1/1
    e2 = conv_block(layers.MaxPool2D((2,2))(e1), base_filters * 2)  #1/2
    e3 = conv_block(layers.MaxPool2D((2,2))(e2), base_filters * 4)  #1/4
    e4 = conv_block(layers.MaxPool2D((2,2))(e3), base_filters * 8)  #1/8
    e5 = conv_block(layers.MaxPool2D((2,2))(e4), base_filters * 16) #1/16

    # Decoder (full-scale aggregation; 5 sources per stage)
    d4 = decoder_stage([pooling(e1, 8), pooling(e2, 4), pooling(e3, 2), e4, upsample(e5, 2)], decoder_filters)      #1/8
    d3 = decoder_stage([pooling(e1, 4), pooling(e2, 2), e3, upsample(d4, 2), upsample(e5, 4)], decoder_filters)     #1/4
    d2 = decoder_stage([pooling(e1, 2), e2, upsample(d3, 2), upsample(d4, 4), upsample(e5, 8)], decoder_filters)    #1/2
    d1 = decoder_stage([e1, upsample(d2, 2), upsample(d3, 4), upsample(d4, 8), upsample(e5, 16)], decoder_filters)  #1/1


    outputs = layers.Conv2D(1, 1)(d1)
    outputs = layers.Activation("sigmoid")(outputs)

    return Model(inputs=inputs, outputs=outputs)



def main() -> None:
    """Build the model and print its summary."""
    model = unet3_plus((512, 512, 3))
    model.summary()

if __name__ == "__main__":
    main()