import tensorflow as tf
print(tf.__version__)


# The pix2pix generator is based on U-net architecture with downsampling and upsampling
OUTPUT_CHANNELS = 3

# Define the downsample layers
def downsample(filters, size, apply_batchnorm=True):
    """
    Creates a downsampling block consisting of Convolutional, BatchNormalization, and LeakyReLU layers.

    Parameters:
    - filters: Number of filters in the Conv2D layer.
    - size: Size of the convolutional kernel.
    - apply_batchnorm: Whether to apply BatchNormalization.

    Returns:
    A sequential model representing the downsampling block.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Define the upsampling layers
def upsample(filters, size, apply_dropout=False):
    """
    Creates an upsampling block consisting of Conv2DTranspose, BatchNormalization, Dropout, and ReLU layers.

    Parameters:
    - filters: Number of filters in the Conv2DTranspose layer.
    - size: Size of the convolutional kernel.
    - apply_dropout: Whether to apply Dropout.

    Returns:
    A sequential model representing the upsampling block.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

def load_image(image_file, height=256, width=256):
    if tf.is_tensor(image_file):
        # If image_file is already a tensor, assume it's the image data
        image = image_file
    else:
        # If image_file is a string, read the file content
        image = tf.io.read_file(tf.convert_to_tensor(image_file, dtype=tf.string))
        image = tf.image.decode_png(image, channels=3)  # Specify the number of channels in the image

    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.float32)  # Convert to float32
    image = (image / 127.5) - 1
    return image

# Specify the complete path to the directory where the SavedModel is stored
model_path = 'Checkpoint'

# Load the model
new_generator = tf.keras.models.load_model(model_path, compile=False)

def generate_images(test_input, model=new_generator):
    # Load the input image
    image_to_load = load_image(test_input)

    # Check the number of channels in the input image
    num_channels = tf.shape(image_to_load)[-1]

    if num_channels == 1:
        # If the input image is grayscale, convert it to RGB
        rgb_image = tf.image.grayscale_to_rgb(image_to_load)
    elif num_channels == 3:
        # If the input image already has 3 channels (RGB), no need to convert
        rgb_image = image_to_load
    elif num_channels == 4:
        # If the input image has 4 channels, remove the alpha channel
        rgb_image = image_to_load[:, :, :3]
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1, 3, or 4.")

    # Expand the dimensions to match the expected input shape (1, 256, 256, 3)
    expanded_image = tf.expand_dims(rgb_image, axis=0)

    # Generate the image using the model
    generated_image = model(expanded_image)

    return generated_image[0]

print("Hello world")
